import argparse
import gzip
import json
import xml.etree.ElementTree as ET
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    with path.open("w", encoding="utf-8") as output:
        for filename in args.xml:
            opener = gzip.open if filename.endswith(".gz") else open
            with opener(filename, "rb") as source:
                for _, element in ET.iterparse(source, events=("end",)):
                    if element.tag != "PubmedArticle":
                        continue
                    pmid = element.findtext("./MedlineCitation/PMID")
                    title = element.find("./MedlineCitation/Article/ArticleTitle")
                    sections = element.findall("./MedlineCitation/Article/Abstract/AbstractText")
                    abstract = " ".join("".join(s.itertext()).strip() for s in sections)
                    if pmid and abstract and pmid not in seen:
                        seen.add(pmid)
                        output.write(json.dumps({"id": "PMID:" + pmid, "text": abstract,
                                                 "title": "".join(title.itertext()) if title is not None else ""}) + "\n")
                    element.clear()
    print(f"Imported {len(seen)} abstracts")


if __name__ == "__main__":
    main()
