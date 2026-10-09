import argparse
import json
import os
import time
from pathlib import Path

import requests

from descfuse.data import read_records
from descfuse.references import BM25, digest, reference_prompt, validate_reference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--codes", required=True)
    parser.add_argument("--umls", required=True)
    parser.add_argument("--abstracts", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--taxonomy", required=True)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--endpoint")
    parser.add_argument("--responses-dir")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()
    if not args.endpoint and not args.responses_dir:
        parser.error("Provide --endpoint or --responses-dir containing cached model responses.")
    allowed_code_fields = {"code", "description"}
    codes = read_records(args.codes)
    if any(set(row) - allowed_code_fields for row in codes):
        raise ValueError("Code input must contain only code and description fields.")
    umls = json.loads(Path(args.umls).read_text(encoding="utf-8"))
    documents = read_records(args.abstracts)
    if any(set(row) - {"id", "text", "title"} for row in documents):
        raise ValueError("Abstract corpus may contain only id, text, and title.")
    for evidence in umls.values():
        if set(evidence) - {"synonyms", "related_concepts", "source_ids"}:
            raise ValueError("UMLS records may contain only synonyms, related_concepts, and source_ids.")
    retriever = BM25(documents)
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    cache = destination / "cache"
    cache.mkdir(exist_ok=True)
    rows = []
    for code in codes:
        evidence = umls.get(code["code"], {"synonyms": [], "related_concepts": [], "source_ids": []})
        query = " ".join([code["description"], *evidence["synonyms"], *evidence["related_concepts"]])
        abstracts = retriever.search(query, args.top_k)
        metadata = {"dataset": args.dataset, "taxonomy": args.taxonomy, "description": code["description"],
                    "code": code["code"], "snapshot": args.snapshot, "umls": evidence, "abstracts": abstracts,
                    "model": args.model, "temperature": args.temperature, "prompt_version": 1}
        key = digest(metadata)
        cache_path = cache / (key + ".json")
        if cache_path.exists():
            saved = json.loads(cache_path.read_text(encoding="utf-8"))
        else:
            messages = reference_prompt(code, evidence, abstracts)
            if args.responses_dir:
                raw = json.loads((Path(args.responses_dir) / (code["code"] + ".json")).read_text(encoding="utf-8"))
                response = raw.get("facets") and raw
                if not response:
                    response = json.loads(raw["choices"][0]["message"]["content"])
            else:
                headers = {"Content-Type": "application/json"}
                if os.environ.get("REFERENCE_API_KEY"):
                    headers["Authorization"] = "Bearer " + os.environ["REFERENCE_API_KEY"]
                payload = {"model": args.model, "messages": messages, "temperature": args.temperature,
                           "response_format": {"type": "json_object"}}
                for attempt in range(3):
                    result = requests.post(args.endpoint, headers=headers, json=payload, timeout=180)
                    if result.status_code not in {429, 500, 502, 503, 504}:
                        break
                    time.sleep(min(2 ** attempt, 4))
                result.raise_for_status()
                raw = result.json()
                response = json.loads(raw["choices"][0]["message"]["content"])
            saved = {"cache_key": key, "metadata": metadata, "messages": messages,
                     "raw_response": raw, "response": response}
            allowed = {"canonical:" + code["code"], *evidence["source_ids"], *[a["id"] for a in abstracts]}
            validate_reference(response, allowed, code["code"])
            cache_path.write_text(json.dumps(saved, ensure_ascii=False, indent=2), encoding="utf-8")
        allowed = {"canonical:" + code["code"], *evidence["source_ids"], *[a["id"] for a in abstracts]}
        facets = validate_reference(saved["response"], allowed, code["code"])
        rows.append({**code, "facets": facets, "cache_key": key, "provenance": "evidence_distilled_reference"})
        print(code["code"], key, flush=True)
    (destination / "codes.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
