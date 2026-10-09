import argparse
import json
from collections import defaultdict
from pathlib import Path

from descfuse.data import read_records


def normalize(code):
    return code.replace(".", "").upper()


def rrf_rows(path):
    with Path(path).open(encoding="utf-8", errors="strict") as handle:
        for line in handle:
            yield line.rstrip("\n").split("|")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--codes", required=True)
    parser.add_argument("--mrconso", required=True)
    parser.add_argument("--mrrel", required=True)
    parser.add_argument("--icd-source", default="ICD9CM", choices=["ICD9CM", "ICD10CM"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-synonyms", type=int, default=20)
    parser.add_argument("--max-related", type=int, default=20)
    args = parser.parse_args()
    codes = read_records(args.codes)
    lookup = {normalize(row["code"]): row["code"] for row in codes}
    code_cuis = defaultdict(set)
    for row in rrf_rows(args.mrconso):
        if row[1] == "ENG" and row[11] == args.icd_source and row[16] == "N" and normalize(row[13]) in lookup:
            code_cuis[lookup[normalize(row[13])]].add(row[0])
    anchors = set().union(*code_cuis.values()) if code_cuis else set()
    neighbors = defaultdict(set)
    relations = {"associated_with", "has_finding_site", "finding_site_of", "has_associated_morphology",
                 "associated_morphology_of", "causes", "caused_by", "has_manifestation", "manifestation_of"}
    for row in rrf_rows(args.mrrel):
        if row[14] != "N" or row[7] not in relations:
            continue
        if row[0] in anchors:
            neighbors[row[0]].add(row[4])
        if row[4] in anchors:
            neighbors[row[4]].add(row[0])
    retained_neighbors = {cui: sorted(related)[:args.max_related] for cui, related in neighbors.items()}
    needed = anchors | {cui for values in retained_neighbors.values() for cui in values}
    terms = defaultdict(set)
    for row in rrf_rows(args.mrconso):
        if row[0] in needed and row[1] == "ENG" and row[16] == "N":
            terms[row[0]].add(row[14])
    output = {}
    for code in codes:
        cuis = code_cuis[code["code"]]
        related = {c for cui in cuis for c in retained_neighbors.get(cui, [])} - cuis
        synonyms = sorted({t for cui in cuis for t in terms[cui]} - {code["description"]})[:args.max_synonyms]
        concepts = sorted({t for cui in related for t in sorted(terms[cui])[:1]})[:args.max_related]
        output[code["code"]] = {"synonyms": synonyms, "related_concepts": concepts,
                                "source_ids": ["UMLS:" + cui for cui in sorted(cuis | related)]}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
