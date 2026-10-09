import argparse
import json
from pathlib import Path
from xml.sax.saxutils import escape


def main():
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="sample/profiles.json")
    parser.add_argument("--output", default="sample/profiles.pdf")
    args = parser.parse_args()
    profiles = json.loads(Path(args.input).read_text(encoding="utf-8"))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(str(output), pagesize=A4, rightMargin=22, leftMargin=22, topMargin=22, bottomMargin=22)
    normal = ParagraphStyle("body", fontName="Helvetica", fontSize=7.8, leading=9.6, textColor=colors.HexColor("#203344"))
    small = ParagraphStyle("small", parent=normal, fontSize=6.8, leading=8.4)
    heading = ParagraphStyle("heading", parent=normal, fontName="Helvetica-Bold", fontSize=14, leading=17, spaceAfter=6)
    white = ParagraphStyle("white", parent=normal, fontName="Helvetica-Bold", textColor=colors.white)
    story = []
    for index, profile in enumerate(profiles):
        if index:
            story.append(PageBreak())
        story.extend([Paragraph("ICD-9 " + profile["code"], heading), Paragraph(escape(profile["description"]), heading), Spacer(1, 4)])
        rows = [[Paragraph("Dimension", white), Paragraph("Direct-Prompting", white), Paragraph("DescFuse", white)]]
        for facet, text in profile["descfuse"].items():
            rows.append([Paragraph(facet, normal), Paragraph(escape(profile["direct_prompting"][facet]), normal),
                         Paragraph(escape(text), normal)])
        table = Table(rows, colWidths=[68, 230, A4[0] - 44 - 68 - 230], repeatRows=1, hAlign="LEFT")
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#244e68")), ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LINEBELOW", (0, 0), (-1, -1), 0.4, colors.HexColor("#d9e2e9")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f3f7fa"), colors.white])]))
        story.extend([table, Spacer(1, 8), Paragraph(escape(profile["coding_note"]), small), Spacer(1, 5)])
        links = [f'<link href="{escape(s["url"])}">{escape(s["title"])}</link>' for s in profile["sources"]]
        story.append(Paragraph("Sources: " + "; ".join(links), small))

    def footer(canvas, doc):
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#617180"))
        canvas.drawRightString(A4[0] - 32, 20, str(doc.page))

    document.build(story, onFirstPage=footer, onLaterPages=footer)
    print(output)


if __name__ == "__main__":
    main()
