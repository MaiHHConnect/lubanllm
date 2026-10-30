"""Local text extraction for teacher source files; never invokes the teacher."""
from html.parser import HTMLParser
from pathlib import Path
import subprocess
import io
import zipfile

from .vision import extract_document, MAX_DOCUMENT_BYTES

DOCUMENT_EXTENSIONS = {".pdf", ".doc", ".docx", ".rtf", ".odt", ".html", ".htm"}
MAX_SOURCE_BYTES = 65536


class _HTMLText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts, self.ignored = [], []

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "noscript", "template"}:
            self.ignored.append(tag)
        elif not self.ignored and tag in {"p", "div", "br", "tr", "li", "h1", "h2", "h3", "section"}:
            self.parts.append("\n")
        elif not self.ignored and tag in {"td", "th"}:
            self.parts.append("\t")

    def handle_endtag(self, tag):
        if self.ignored and tag == self.ignored[-1]:
            self.ignored.pop()
        elif not self.ignored and tag in {"p", "div", "tr", "li", "section"}:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.ignored:
            self.parts.append(data)


def extract_source(raw, name):
    if not isinstance(name, str) or not name or len(name) > 300 or "\x00" in name:
        raise ValueError("附件文件名无效")
    suffix = Path(name).suffix.lower()
    if suffix not in DOCUMENT_EXTENSIONS:
        raise ValueError("不支持此格式")
    if not raw or len(raw) > MAX_DOCUMENT_BYTES:
        raise ValueError("文档为空或超过 8 MiB")
    if suffix == ".docx":
        text = extract_document(raw, name)["text"]
    elif suffix in {".html", ".htm"}:
        try:
            parser = _HTMLText()
            parser.feed(raw.decode("utf-8-sig"))
            parser.close()
            text = "".join(parser.parts)
        except (UnicodeError, ValueError) as error:
            raise ValueError("HTML 需要使用 UTF-8 编码") from error
    else:
        if suffix == ".odt":
            try:
                with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                    entries = archive.infolist()
                    if len(entries) > 1024 or sum(item.file_size for item in entries) > 16 * 1024 * 1024:
                        raise ValueError("文档解压后过大，请拆分文件后重试")
                    if any(item.flag_bits & 1 for item in entries):
                        raise ValueError("加密文档不能读取")
            except zipfile.BadZipFile as error:
                raise ValueError("文档无法读取") from error
        root = Path(__file__).resolve().parents[2]
        helper = next((p for p in (root / "native/luban-document-text", root / "build/native/luban-document-text")
                       if p.is_file() and not p.is_symlink()), None)
        if helper is None:
            raise ValueError("文档解析器未就绪，请更新应用")
        try:
            result = subprocess.run([str(helper), suffix[1:]], input=raw, capture_output=True, timeout=30)
        except subprocess.TimeoutExpired as error:
            raise ValueError("文档解析超时，请拆分文件后重试") from error
        if result.returncode == 2:
            raise ValueError("提取文字超过 64 KiB，请拆分文件后重试")
        if result.returncode != 0:
            raise ValueError("文档无法读取：请检查是否损坏、加密或为无文字的扫描件")
        text = result.stdout.decode("utf-8")
    text = text.replace("\x0c", "\n").strip()
    if not text:
        raise ValueError("文档没有可提取文字，扫描件请先转为文字")
    if len(text.encode("utf-8")) > MAX_SOURCE_BYTES:
        raise ValueError("提取文字超过 64 KiB，请拆分文件后重试")
    return {"text": text, "format": suffix[1:]}
