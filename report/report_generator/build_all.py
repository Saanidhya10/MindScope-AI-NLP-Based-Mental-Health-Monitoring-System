# build_all.py
"""
Compiles the full humanized Minor Project Report into:
1. Humanized_Minor_Project_Report.docx (Word Document)
2. Humanized_Minor_Project_Report.md (Markdown Document)
3. Humanized_Minor_Project_Report.pdf (PDF Document)
"""

import os
import sys

# Ensure report_generator is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import front_matter
import chapter1
import chapter2
import chapter3
import chapter4
import chapter5
import chapter6
import chapter7
import chapter8

import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT

WORKSPACE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), fill_hex)
    tcPr.append(shd)

def create_docx():
    print("Building DOCX report...")
    doc = docx.Document()
    
    # Page setup (margins: left 1.25 in for binding, 1.0 in others)
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.25)
        section.right_margin = Inches(1.0)
    
    # Base styling
    normal_style = doc.styles['Normal']
    normal_style.font.name = 'Times New Roman'
    normal_style.font.size = Pt(12)
    normal_style.font.color.rgb = RGBColor(30, 30, 30)
    normal_style.paragraph_format.line_spacing = 1.15
    normal_style.paragraph_format.space_after = Pt(6)
    
    # Title Page
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_before = Pt(36)
    p_title.paragraph_format.space_after = Pt(12)
    run_title = p_title.add_run(front_matter.TITLE_DATA["title"])
    run_title.font.name = 'Times New Roman'
    run_title.font.size = Pt(22)
    run_title.bold = True
    run_title.font.color.rgb = RGBColor(16, 44, 87)
    
    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_sub.paragraph_format.space_after = Pt(36)
    r_sub = p_sub.add_run("A Minor Project Report Submitted in Partial Fulfilment of the Requirements for the Degree of\n" + front_matter.TITLE_DATA["degree"])
    r_sub.font.name = 'Times New Roman'
    r_sub.font.size = Pt(12)
    r_sub.italic = True
    
    p_inst = doc.add_paragraph()
    p_inst.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_inst.paragraph_format.space_after = Pt(48)
    r_inst = p_inst.add_run(front_matter.TITLE_DATA["department"] + "\n" + front_matter.TITLE_DATA["institution"] + "\n\n" + front_matter.TITLE_DATA["academic_year"])
    r_inst.font.name = 'Times New Roman'
    r_inst.font.size = Pt(12)
    r_inst.bold = True
    
    p_team = doc.add_paragraph()
    p_team.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_team = p_team.add_run("Submitted by:\nUtkarsh & Project Team\nDepartment of AI-DS, USAR\n\nUnder the Supervision of:\nFaculty Supervisor, Department of AI-DS")
    r_team.font.name = 'Times New Roman'
    r_team.font.size = Pt(12)
    
    doc.add_page_break()
    
    # Certificate
    h_cert = doc.add_heading("CERTIFICATE", level=1)
    h_cert.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for para in front_matter.CERTIFICATE_TEXT.split("\n\n"):
        doc.add_paragraph(para.strip())
    doc.add_page_break()
    
    # Declaration
    h_decl = doc.add_heading("DECLARATION", level=1)
    h_decl.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for para in front_matter.DECLARATION_TEXT.split("\n\n"):
        doc.add_paragraph(para.strip())
    doc.add_page_break()
    
    # Abstract
    h_abs = doc.add_heading("ABSTRACT", level=1)
    h_abs.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for para in front_matter.ABSTRACT_TEXT.split("\n\n"):
        doc.add_paragraph(para.strip())
    doc.add_page_break()
    
    # Acknowledgements
    h_ack = doc.add_heading("ACKNOWLEDGEMENTS", level=1)
    h_ack.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for para in front_matter.ACKNOWLEDGEMENTS_TEXT.split("\n\n"):
        doc.add_paragraph(para.strip())
    doc.add_page_break()
    
    # Helper for adding chapters
    def add_chapter(title, sections, tables=None):
        h = doc.add_heading(title, level=1)
        h.paragraph_format.space_before = Pt(12)
        h.paragraph_format.space_after = Pt(12)
        h.runs[0].font.name = 'Times New Roman'
        h.runs[0].font.size = Pt(16)
        h.runs[0].bold = True
        h.runs[0].font.color.rgb = RGBColor(16, 44, 87)
        
        for sec_title, sec_body in sections:
            sh = doc.add_heading(sec_title, level=2)
            sh.paragraph_format.space_before = Pt(10)
            sh.paragraph_format.space_after = Pt(4)
            sh.runs[0].font.name = 'Times New Roman'
            sh.runs[0].font.size = Pt(13)
            sh.runs[0].bold = True
            sh.runs[0].font.color.rgb = RGBColor(30, 30, 30)
            
            for p in sec_body.split("\n\n"):
                doc.add_paragraph(p.strip())
                
        if tables:
            for tbl_info in tables:
                p_tbl_title = doc.add_paragraph()
                p_tbl_title.paragraph_format.space_before = Pt(12)
                p_tbl_title.paragraph_format.space_after = Pt(4)
                r_tbl_title = p_tbl_title.add_run(tbl_info["title"])
                r_tbl_title.bold = True
                r_tbl_title.font.name = 'Times New Roman'
                r_tbl_title.font.size = Pt(11)
                
                rows_count = len(tbl_info["rows"]) + 1
                cols_count = len(tbl_info["headers"])
                table = doc.add_table(rows=rows_count, cols=cols_count)
                table.alignment = WD_TABLE_ALIGNMENT.CENTER
                table.autofit = True
                
                # Header row
                hdr_cells = table.rows[0].cells
                for i, h_text in enumerate(tbl_info["headers"]):
                    hdr_cells[i].text = h_text
                    set_cell_background(hdr_cells[i], '102C57')
                    for p in hdr_cells[i].paragraphs:
                        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                        for r in p.runs:
                            r.bold = True
                            r.font.name = 'Times New Roman'
                            r.font.size = Pt(10)
                            r.font.color.rgb = RGBColor(255, 255, 255)
                            
                # Data rows
                for r_idx, row_data in enumerate(tbl_info["rows"]):
                    row_cells = table.rows[r_idx + 1].cells
                    for c_idx, val in enumerate(row_data):
                        row_cells[c_idx].text = str(val)
                        if r_idx % 2 == 1:
                            set_cell_background(row_cells[c_idx], 'F5F7FA')
                        for p in row_cells[c_idx].paragraphs:
                            p.paragraph_format.space_after = Pt(2)
                            for r in p.runs:
                                r.font.name = 'Times New Roman'
                                r.font.size = Pt(9.5)
                doc.add_paragraph()
        doc.add_page_break()

    add_chapter(chapter1.CHAPTER_1_TITLE, chapter1.SECTIONS)
    add_chapter(chapter2.CHAPTER_2_TITLE, chapter2.SECTIONS, [chapter2.TABLE_2_1_DATA])
    add_chapter(chapter3.CHAPTER_3_TITLE, chapter3.SECTIONS)
    add_chapter(chapter4.CHAPTER_4_TITLE, chapter4.SECTIONS)
    add_chapter(chapter5.CHAPTER_5_TITLE, chapter5.SECTIONS, [chapter5.TABLE_5_1_DATA, chapter5.TABLE_5_2_DATA])
    add_chapter(chapter6.CHAPTER_6_TITLE, chapter6.SECTIONS, [chapter6.TABLE_6_1_DATA, chapter6.TABLE_6_2_DATA, chapter6.TABLE_6_3_DATA])
    add_chapter(chapter7.CHAPTER_7_TITLE, chapter7.SECTIONS, [chapter7.TABLE_7_1_DATA])
    
    # Chapter 8: References
    h_ref = doc.add_heading(chapter8.CHAPTER_8_TITLE, level=1)
    h_ref.runs[0].font.name = 'Times New Roman'
    h_ref.runs[0].font.size = Pt(16)
    h_ref.runs[0].bold = True
    h_ref.runs[0].font.color.rgb = RGBColor(16, 44, 87)
    
    for ref in chapter8.REFERENCES_LIST:
        p_ref = doc.add_paragraph(ref)
        p_ref.paragraph_format.left_indent = Inches(0.5)
        p_ref.paragraph_format.first_line_indent = Inches(-0.5)
        p_ref.paragraph_format.space_after = Pt(4)
        for r in p_ref.runs:
            r.font.name = 'Times New Roman'
            r.font.size = Pt(10.5)
            
    docx_path = os.path.join(WORKSPACE_DIR, "Humanized_Minor_Project_Report.docx")
    doc.save(docx_path)
    print(f"DOCX report saved: {docx_path} ({os.path.getsize(docx_path):,} bytes)")
    return docx_path

def create_markdown():
    print("Building Markdown report...")
    lines = []
    
    lines.append(f"# {front_matter.TITLE_DATA['title']}\n")
    lines.append(f"**{front_matter.TITLE_DATA['degree']}**  \n")
    lines.append(f"{front_matter.TITLE_DATA['department']}  \n")
    lines.append(f"{front_matter.TITLE_DATA['institution']}  \n\n")
    lines.append(f"**Author:** {front_matter.TITLE_DATA['author']}  \n")
    lines.append(f"**Academic Year:** {front_matter.TITLE_DATA['academic_year']}  \n\n---\n")
    
    lines.append("## ABSTRACT\n")
    lines.append(front_matter.ABSTRACT_TEXT + "\n\n---\n")
    
    def render_md_table(tbl_data):
        out = [f"\n### {tbl_data['title']}\n\n"]
        headers = tbl_data['headers']
        out.append("| " + " | ".join(headers) + " |\n")
        out.append("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in tbl_data['rows']:
            out.append("| " + " | ".join(str(c).replace("\n", " ") for c in row) + " |\n")
        out.append("\n")
        return "".join(out)
        
    def add_md_chapter(title, sections, tables=None):
        res = [f"# {title}\n\n"]
        for sec_title, sec_body in sections:
            res.append(f"## {sec_title}\n\n{sec_body}\n\n")
        if tables:
            for tbl in tables:
                res.append(render_md_table(tbl))
        res.append("---\n\n")
        return "".join(res)
        
    lines.append(add_md_chapter(chapter1.CHAPTER_1_TITLE, chapter1.SECTIONS))
    lines.append(add_md_chapter(chapter2.CHAPTER_2_TITLE, chapter2.SECTIONS, [chapter2.TABLE_2_1_DATA]))
    lines.append(add_md_chapter(chapter3.CHAPTER_3_TITLE, chapter3.SECTIONS))
    lines.append(add_md_chapter(chapter4.CHAPTER_4_TITLE, chapter4.SECTIONS))
    lines.append(add_md_chapter(chapter5.CHAPTER_5_TITLE, chapter5.SECTIONS, [chapter5.TABLE_5_1_DATA, chapter5.TABLE_5_2_DATA]))
    lines.append(add_md_chapter(chapter6.CHAPTER_6_TITLE, chapter6.SECTIONS, [chapter6.TABLE_6_1_DATA, chapter6.TABLE_6_2_DATA, chapter6.TABLE_6_3_DATA]))
    lines.append(add_md_chapter(chapter7.CHAPTER_7_TITLE, chapter7.SECTIONS, [chapter7.TABLE_7_1_DATA]))
    
    lines.append(f"# {chapter8.CHAPTER_8_TITLE}\n\n")
    for ref in chapter8.REFERENCES_LIST:
        lines.append(f"- {ref}\n")
        
    md_path = os.path.join(WORKSPACE_DIR, "Humanized_Minor_Project_Report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("".join(lines))
    print(f"Markdown report saved: {md_path} ({os.path.getsize(md_path):,} bytes)")
    return md_path

def create_pdf():
    print("Building PDF report via ReportLab...")
    pdf_path = os.path.join(WORKSPACE_DIR, "Humanized_Minor_Project_Report.pdf")
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )
    
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#102C57'),
        alignment=TA_CENTER,
        spaceAfter=15
    )
    
    h1_style = ParagraphStyle(
        'DocH1',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=15,
        leading=18,
        textColor=colors.HexColor('#102C57'),
        spaceBefore=14,
        spaceAfter=8,
        keepWithNext=True
    )
    
    h2_style = ParagraphStyle(
        'DocH2',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=15,
        textColor=colors.HexColor('#222222'),
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'DocBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#2B2B2B'),
        alignment=TA_JUSTIFY,
        spaceAfter=6
    )
    
    caption_style = ParagraphStyle(
        'DocCaption',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9.5,
        leading=12,
        textColor=colors.HexColor('#102C57'),
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True
    )
    
    cell_style = ParagraphStyle(
        'DocCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=10,
        textColor=colors.HexColor('#222222')
    )
    
    cell_hdr_style = ParagraphStyle(
        'DocCellHdr',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=11,
        textColor=colors.white,
        alignment=TA_CENTER
    )
    
    ref_style = ParagraphStyle(
        'DocRef',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor('#2B2B2B'),
        leftIndent=15,
        firstLineIndent=-15,
        spaceAfter=4
    )
    
    story = []
    
    # Title Page
    story.append(Spacer(1, 40))
    story.append(Paragraph(front_matter.TITLE_DATA["title"], title_style))
    story.append(Spacer(1, 10))
    story.append(Paragraph(f"<b>{front_matter.TITLE_DATA['degree']}</b>", ParagraphStyle('Sub1', parent=body_style, alignment=TA_CENTER)))
    story.append(Spacer(1, 15))
    story.append(Paragraph(f"{front_matter.TITLE_DATA['department']}<br/>{front_matter.TITLE_DATA['institution']}<br/><br/><b>{front_matter.TITLE_DATA['academic_year']}</b>", ParagraphStyle('Sub2', parent=body_style, alignment=TA_CENTER)))
    story.append(Spacer(1, 30))
    story.append(Paragraph("<b>Submitted by:</b><br/>Utkarsh & Project Team<br/>Department of AI-DS, USAR<br/><br/><b>Under the Supervision of:</b><br/>Faculty Supervisor, Department of AI-DS", ParagraphStyle('Sub3', parent=body_style, alignment=TA_CENTER)))
    story.append(PageBreak())
    
    # Certificate & Declaration
    story.append(Paragraph("CERTIFICATE", h1_style))
    for p in front_matter.CERTIFICATE_TEXT.split("\n\n"):
        story.append(Paragraph(p.strip().replace("\n", "<br/>"), body_style))
    story.append(PageBreak())
    
    story.append(Paragraph("DECLARATION", h1_style))
    for p in front_matter.DECLARATION_TEXT.split("\n\n"):
        story.append(Paragraph(p.strip().replace("\n", "<br/>"), body_style))
    story.append(PageBreak())
    
    # Abstract
    story.append(Paragraph("ABSTRACT", h1_style))
    for p in front_matter.ABSTRACT_TEXT.split("\n\n"):
        story.append(Paragraph(p.strip().replace("\n", "<br/>"), body_style))
    story.append(PageBreak())
    
    def make_rl_table(tbl_data):
        headers = [Paragraph(f"<b>{h}</b>", cell_hdr_style) for h in tbl_data['headers']]
        data = [headers]
        for row in tbl_data['rows']:
            data.append([Paragraph(str(c).replace("\n", "<br/>"), cell_style) for c in row])
            
        col_count = len(tbl_data['headers'])
        available_width = letter[0] - 108  # margins 54 left, 54 right
        col_width = available_width / col_count
        
        t = Table(data, colWidths=[col_width]*col_count)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#102C57')),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#F8FAFC')]),
            ('TOPPADDING', (0,0), (-1,-1), 4),
            ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ]))
        return t

    def add_pdf_chapter(title, sections, tables=None):
        story.append(Paragraph(title, h1_style))
        story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#102C57'), spaceAfter=10))
        for sec_title, sec_body in sections:
            story.append(Paragraph(sec_title, h2_style))
            for p in sec_body.split("\n\n"):
                story.append(Paragraph(p.strip().replace("\n", "<br/>"), body_style))
        if tables:
            for tbl in tables:
                story.append(Spacer(1, 6))
                story.append(Paragraph(f"<b>{tbl['title']}</b>", caption_style))
                story.append(make_rl_table(tbl))
                story.append(Spacer(1, 8))
        story.append(PageBreak())

    add_pdf_chapter(chapter1.CHAPTER_1_TITLE, chapter1.SECTIONS)
    add_pdf_chapter(chapter2.CHAPTER_2_TITLE, chapter2.SECTIONS, [chapter2.TABLE_2_1_DATA])
    add_pdf_chapter(chapter3.CHAPTER_3_TITLE, chapter3.SECTIONS)
    add_pdf_chapter(chapter4.CHAPTER_4_TITLE, chapter4.SECTIONS)
    add_pdf_chapter(chapter5.CHAPTER_5_TITLE, chapter5.SECTIONS, [chapter5.TABLE_5_1_DATA, chapter5.TABLE_5_2_DATA])
    add_pdf_chapter(chapter6.CHAPTER_6_TITLE, chapter6.SECTIONS, [chapter6.TABLE_6_1_DATA, chapter6.TABLE_6_2_DATA, chapter6.TABLE_6_3_DATA])
    add_pdf_chapter(chapter7.CHAPTER_7_TITLE, chapter7.SECTIONS, [chapter7.TABLE_7_1_DATA])
    
    # Chapter 8
    story.append(Paragraph(chapter8.CHAPTER_8_TITLE, h1_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#102C57'), spaceAfter=10))
    for ref in chapter8.REFERENCES_LIST:
        story.append(Paragraph(ref, ref_style))
        
    doc.build(story)
    print(f"PDF report saved: {pdf_path} ({os.path.getsize(pdf_path):,} bytes)")
    return pdf_path

if __name__ == "__main__":
    docx_file = create_docx()
    md_file = create_markdown()
    pdf_file = create_pdf()
    print("\nAll formats compiled successfully!")
