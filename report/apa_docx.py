"""Helpers that write content into the provided APA 7 template without touching its styles.

The template formats everything with direct paragraph formatting (double spacing,
centred bold headings, 0.5" first-line indents, hanging-indent references), so these
helpers reproduce exactly that formatting instead of introducing new styles.

Inline markup supported in text: *italic* and **bold**.
"""
import copy
import re

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_LINE_SPACING
from docx.shared import Inches, Pt

_TOKEN = re.compile(r"(\*\*[^*]+\*\*|\*[^*]+\*)")


class APADoc:
    def __init__(self, template_path):
        self.doc = Document(template_path)
        body = self.doc.element.body
        self._sectPr = copy.deepcopy(body.sectPr)
        for child in list(body):           # clear the template's placeholder content
            body.remove(child)
        body.append(self._sectPr)

    # -- low level -------------------------------------------------------
    def _para(self, align=None, first_line=None, hanging=False, double=True, keep_next=False):
        p = self.doc.add_paragraph()
        pf = p.paragraph_format
        if double:
            pf.line_spacing_rule = WD_LINE_SPACING.DOUBLE
        pf.space_before = Pt(0)
        pf.space_after = Pt(0)
        if align == "center":
            pf.alignment = WD_ALIGN_PARAGRAPH.CENTER
        if first_line:
            pf.first_line_indent = Inches(first_line)
        if hanging:
            pf.left_indent = Inches(0.5)
            pf.first_line_indent = Inches(-0.5)
        if keep_next:
            pf.keep_with_next = True
        return p

    @staticmethod
    def _runs(p, text, bold=False, italic=False, size=None):
        for part in _TOKEN.split(text):
            if not part:
                continue
            b, i, t = bold, italic, part
            if part.startswith("**") and part.endswith("**"):
                b, t = True, part[2:-2]
            elif part.startswith("*") and part.endswith("*"):
                i, t = True, part[1:-1]
            r = p.add_run(t)
            r.bold, r.italic = b or None, i or None
            if size:
                r.font.size = Pt(size)
        return p

    # -- building blocks --------------------------------------------------
    def title_page(self, title, lines):
        p = self._para(align="center")
        p.paragraph_format.space_before = Pt(180)   # template: 3600 twips before the title
        self._runs(p, title, bold=True)
        self._para(align="center")
        for line in lines:
            self._runs(self._para(align="center"), line)
        self.page_break()

    def page_break(self):
        self._para().add_run().add_break(WD_BREAK.PAGE)

    def title(self, text):
        self._runs(self._para(align="center", keep_next=True), text, bold=True)

    def heading1(self, text):
        self._runs(self._para(align="center", keep_next=True), text, bold=True)

    def heading2(self, text):
        self._runs(self._para(keep_next=True), text, bold=True)

    def body(self, text):
        self._runs(self._para(first_line=0.5), text)

    def reference(self, text):
        self._runs(self._para(hanging=True), text)

    def figure(self, number, title, image_path, width_in=6.0, note=None):
        self._runs(self._para(keep_next=True), f"Figure {number}", bold=True)
        self._runs(self._para(keep_next=True), title, italic=True)
        p = self._para(align="center", double=False, keep_next=bool(note))
        p.add_run().add_picture(str(image_path), width=Inches(width_in))
        if note:
            self._runs(self._para(), f"*Note.* {note}")

    def table(self, number, title, header, rows, col_widths_in, note=None):
        self._runs(self._para(keep_next=True), f"Table {number}", bold=True)
        self._runs(self._para(keep_next=True), title, italic=True)
        t = self.doc.add_table(rows=1 + len(rows), cols=len(header))
        t.autofit = False
        for r_i, row in enumerate([header] + rows):
            for c_i, val in enumerate(row):
                cell = t.cell(r_i, c_i)
                cell.width = Inches(col_widths_in[c_i])
                para = cell.paragraphs[0]
                para.paragraph_format.space_after = Pt(0)
                if c_i > 0:
                    para.alignment = WD_ALIGN_PARAGRAPH.CENTER
                self._runs(para, str(val), bold=(r_i == 0), size=10)
        _apa_table_borders(t)
        if note:
            self._runs(self._para(), f"*Note.* {note}")
        else:
            self._para()

    def save(self, path):
        self.doc.save(path)


def _apa_table_borders(table):
    # APA style: horizontal rules above and below the header and at the bottom only.
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    tblPr = table._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge, val in (("top", "single"), ("bottom", "single"), ("left", "nil"), ("right", "nil"),
                      ("insideH", "nil"), ("insideV", "nil")):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), val)
        if val == "single":
            el.set(qn("w:sz"), "8"); el.set(qn("w:space"), "0"); el.set(qn("w:color"), "000000")
        borders.append(el)
    tblPr.append(borders)
    for cell in table.rows[0].cells:
        tcPr = cell._tc.get_or_add_tcPr()
        tcb = OxmlElement("w:tcBorders")
        b = OxmlElement("w:bottom")
        b.set(qn("w:val"), "single"); b.set(qn("w:sz"), "8"); b.set(qn("w:space"), "0"); b.set(qn("w:color"), "000000")
        tcb.append(b)
        tcPr.append(tcb)
