import argparse
import html
import json
import re
from pathlib import Path

from descfuse import FACETS


def latex_escape(text):
    replacements = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
                    "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(replacements.get(char, char) for char in text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="sample/profiles.json")
    parser.add_argument("--output-dir", default="sample")
    args = parser.parse_args()
    samples = json.loads(Path(args.input).read_text(encoding="utf-8"))
    destination = Path(args.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    md = ["# Multi-faceted clinical profiles", ""]
    tex = []
    web = ['<!doctype html><html lang="en"><meta charset="utf-8"><title>DescFuse sample profiles</title>',
           '<style>body{font:15px/1.55 Arial,sans-serif;max-width:1200px;margin:40px auto;padding:0 24px;color:#172b3a}'
           'table{border-collapse:collapse;width:100%;margin:18px 0 34px}td,th{padding:12px;border:1px solid #d7dfe5;'
           'text-align:left;vertical-align:top}th{background:#eaf2f6}h2{margin-top:38px}'
           'th:first-child{width:11%}th:nth-child(2){width:42%}th:nth-child(3){width:47%}'
           '@media print{h2{break-before:page}tr{break-inside:avoid}body{font-size:10pt;margin:0}}</style>',
           '<h1>Multi-faceted clinical profiles</h1>']
    codes = []
    for sample in samples:
        title = f"ICD-9 {sample['code']}: {sample['description']}"
        md += ["## " + title, "", sample.get("coding_note", ""), "",
               "| Dimension | Direct-Prompting | DescFuse |",
               "|---|---|---|"]
        caption = f"Multi-faceted clinical profiles for ICD-9 {sample['code']} ({sample['description']})."
        label = re.sub(r"[^a-zA-Z0-9]", "", sample["code"])
        tex += [r"\begin{table*}[p]", r"\centering", r"\footnotesize", r"\renewcommand{\arraystretch}{1.15}",
                r"\setlength{\tabcolsep}{4pt}", r"\caption{" + latex_escape(caption) + "}",
                r"\label{tab:profile_" + label + "}",
                r"\begin{tabularx}{\textwidth}{>{\raggedright\arraybackslash}p{1.5cm}>{\raggedright\arraybackslash}X>{\raggedright\arraybackslash}X}",
                r"\toprule", r"\textbf{Dimension} & \textbf{Direct-Prompting} & \textbf{DescFuse} \\",
                r"\midrule"]
        web += [f"<h2>{html.escape(title)}</h2><p>{html.escape(sample.get('coding_note',''))}</p>",
                '<table><thead><tr><th>Dimension</th><th>Direct-Prompting</th>'
                '<th>DescFuse</th></tr></thead><tbody>']
        facets = []
        for facet in FACETS:
            direct, detailed = sample["direct_prompting"][facet], sample["descfuse"][facet]
            md.append(f"| {facet} | {direct} | {detailed} |")
            tex.append(latex_escape(facet) + " & " + latex_escape(direct) + " & " + latex_escape(detailed) + r" \\")
            web.append(f'<tr><td>{facet}</td><td>{html.escape(direct)}</td><td>{html.escape(detailed)}</td></tr>')
            facets.append({"name": facet, "description": detailed, "supported": True,
                           "source_ids": sample["facet_sources"][facet]})
        tex += [r"\bottomrule", r"\end{tabularx}", r"\par\vspace{3pt}",
                r"\begin{minipage}{\textwidth}\footnotesize " + latex_escape(sample.get("coding_note", "")) +
                r" Sources: " + "; ".join(r"\href{" + s["url"] + "}{" + latex_escape(s["title"]) + "}" for s in sample["sources"]) +
                r"\end{minipage}",
                r"\end{table*}", ""]
        web.append("</tbody></table><p>Sources: " + ", ".join(
            f'<a href="{html.escape(source["url"], quote=True)}">{html.escape(source["title"])}</a>'
            for source in sample["sources"]) + "</p>")
        md += ["", "Sources: " + "; ".join(f"[{s['title']}]({s['url']})" for s in sample["sources"]), ""]
        codes.append({"code": sample["code"], "description": sample["description"], "facets": facets,
                      "provenance": sample["provenance"]})
    (destination / "profiles.md").write_text("\n".join(md), encoding="utf-8")
    (destination / "profiles.tex").write_text("\n".join(tex), encoding="utf-8")
    (destination / "profiles.html").write_text("\n".join(web) + "</html>", encoding="utf-8")
    (destination / "codes.json").write_text(json.dumps(codes, indent=2), encoding="utf-8")
    document = (r"\documentclass[10pt]{article}" + "\n" + r"\usepackage[a4paper,margin=1.5cm]{geometry}" + "\n" +
                r"\usepackage{booktabs,tabularx,array,xcolor}" + "\n" + r"\usepackage[hidelinks]{hyperref}" + "\n" + r"\begin{document}" + "\n" +
                "\n".join(tex) + "\n" + r"\end{document}" + "\n")
    (destination / "sample_tables.tex").write_text(document, encoding="utf-8")
    print(f"Exported {len(samples)} profiles to {destination}")


if __name__ == "__main__":
    main()
