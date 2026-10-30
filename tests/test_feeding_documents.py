import io
from html.parser import HTMLParser
from pathlib import Path
import subprocess
import tempfile
import unittest
import zipfile

from luban.documents import extract_source

ROOT = Path(__file__).resolve().parents[1]


def zip_document(name, text):
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as archive:
        archive.writestr(name, text)
    return out.getvalue()


def pdf_document():
    stream = b"BT /F1 18 Tf 40 180 Td (Teacher source example) Tj ET"
    objects = [b"<< /Type /Catalog /Pages 2 0 R >>", b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
               b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 240] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
               b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
               b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream"]
    raw, offsets = b"%PDF-1.4\n", []
    for i, obj in enumerate(objects, 1):
        offsets.append(len(raw)); raw += str(i).encode() + b" 0 obj\n" + obj + b"\nendobj\n"
    start = len(raw)
    raw += b"xref\n0 6\n0000000000 65535 f \n" + b"".join(f"{x:010d} 00000 n \n".encode() for x in offsets)
    return raw + f"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n{start}\n%%EOF\n".encode()


class DocumentsTests(unittest.TestCase):
    def test_docx_and_html_extract_visible_text_without_external_resources(self):
        xml = '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>教师原始资料</w:t></w:r></w:p></w:body></w:document>'
        self.assertEqual(extract_source(zip_document("word/document.xml", xml), "教材.docx")["text"], "教师原始资料")
        html = b'<p>First &amp; second</p><script>secret script</script><style>hidden</style><p>Next</p><img src="https://offline.invalid/x">'
        self.assertEqual(extract_source(html, "notes.html")["text"], "First & second\n\nNext")
        for raw, name in [(b"", "x.pdf"), (b"x", "x.exe"), (b"broken", "x.docx"), (b"x" * 65537, "x.html")]:
            with self.subTest(name=name), self.assertRaises(ValueError): extract_source(raw, name)

    @unittest.skipUnless((ROOT / "build/native/luban-document-text").exists(), "Build native document helper first")
    def test_native_pdf_rtf_doc_and_odt(self):
        self.assertIn("Teacher source example", extract_source(pdf_document(), "notes.pdf")["text"])
        raw = b'{\\rtf1\\ansi Teacher source example\\par Another paragraph}'
        self.assertIn("Another paragraph", extract_source(raw, "notes.rtf")["text"])
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "source.rtf"; source.write_bytes(raw)
            for kind in ("doc", "odt"):
                target = Path(folder) / ("source." + kind)
                subprocess.run(["/usr/bin/textutil", "-convert", kind, str(source), "-output", str(target)], check=True, capture_output=True)
                self.assertIn("Teacher source example", extract_source(target.read_bytes(), target.name)["text"])
        for raw, name in [(b"not pdf", "x.pdf"), (b"{\\rtf1 " + b"x" * 65537 + b"}", "big.rtf")]:
            with self.subTest(name=name), self.assertRaises(ValueError): extract_source(raw, name)

    def test_learning_workspace_is_inside_learning_page(self):
        class Structure(HTMLParser):
            def __init__(self):
                super().__init__(); self.stack = []; self.parents = {}
            def handle_starttag(self, tag, attrs):
                ident = dict(attrs).get("id")
                if ident: self.parents[ident] = [i for _, i in self.stack if i]
                if tag not in {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}:
                    self.stack.append((tag, ident))
            def handle_endtag(self, tag):
                for i in range(len(self.stack) - 1, -1, -1):
                    if self.stack[i][0] == tag:
                        self.stack = self.stack[:i]; break
        tree = Structure(); tree.feed((ROOT / "frontend/index.html").read_text())
        self.assertIn("view-learning", tree.parents["feeding-panel"])
        self.assertNotIn("view-learning", tree.parents["view-settings"])
        self.assertNotIn("feeding-panel", tree.parents["view-leaderboard"])


if __name__ == "__main__": unittest.main()
