"""Extract text and metadata from uploaded files. Supported: txt, md, html, pdf, json."""
import io
import json
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import PurePath

from pypdf import PdfReader

SUPPORTED = {"txt", "md", "html", "pdf", "json"}


class ParseError(ValueError):
    pass


@dataclass
class ParsedDoc:
    title: str
    text: str
    doc_type: str
    metadata: dict = field(default_factory=dict)


class _TextExtractor(HTMLParser):
    """Collects visible text; skips script/style; remembers <title>."""
    SKIP = {"script", "style", "head"}
    BLOCK = {"p", "div", "br", "li", "h1", "h2", "h3", "h4", "tr", "section", "article"}

    def __init__(self):
        super().__init__()
        self.parts, self.title, self._skip, self._in_title = [], "", 0, False

    def handle_starttag(self, tag, attrs):
        if tag == "title":
            self._in_title = True
        if tag in self.SKIP:
            self._skip += 1
        if tag in self.BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False
        if tag in self.SKIP and self._skip:
            self._skip -= 1
        if tag in self.BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif not self._skip:
            self.parts.append(data)


def _clean(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _decode(data: bytes) -> str:
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("latin-1")


def _title_from_first_line(text: str, fallback: str) -> tuple[str, str]:
    """Plain text: a short first line followed by more text is the title (and is removed from the body)."""
    first, _, rest = text.partition("\n")
    if first.strip() and len(first) <= 100 and rest.strip():
        return first.strip(), rest.strip()
    return fallback, text


def _title_from_markdown(text: str, fallback: str) -> str:
    m = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
    return m.group(1).strip() if m else fallback


def parse_file(filename: str, data: bytes) -> ParsedDoc:
    """Parse raw bytes by file extension. Raises ParseError on unsupported or unreadable input."""
    p = PurePath(filename)
    ext = p.suffix.lower().lstrip(".")
    if ext == "htm":
        ext = "html"
    if ext not in SUPPORTED:
        raise ParseError(f"unsupported file type: .{ext or '?'}")
    fallback_title = p.stem

    try:
        if ext == "md":
            text = _clean(_decode(data))
            return ParsedDoc(_title_from_markdown(text, fallback_title), text, ext, {"filename": filename})
        if ext == "txt":
            title, text = _title_from_first_line(_clean(_decode(data)), fallback_title)
            return ParsedDoc(title, text, ext, {"filename": filename})
        if ext == "html":
            ex = _TextExtractor()
            ex.feed(_decode(data))
            title = _clean(ex.title) or fallback_title
            return ParsedDoc(title, _clean("".join(ex.parts)), ext, {"filename": filename})
        if ext == "pdf":
            reader = PdfReader(io.BytesIO(data))
            text = _clean("\n\n".join((pg.extract_text() or "") for pg in reader.pages))
            info = reader.metadata
            title = (info.title if info and info.title else None) or fallback_title
            return ParsedDoc(str(title), text, ext, {"filename": filename, "pages": len(reader.pages)})
        obj = json.loads(_decode(data))  # json
        if not isinstance(obj, dict) or "text" not in obj:
            raise ParseError('json documents must be an object with a "text" field')
        meta = {k: v for k, v in obj.items() if k not in ("text", "title")}
        meta["filename"] = filename
        return ParsedDoc(str(obj.get("title") or fallback_title), _clean(str(obj["text"])), ext, meta)
    except ParseError:
        raise
    except Exception as e:  # noqa: BLE001 - corrupt pdf, bad json, etc. all become one error type
        raise ParseError(f"could not parse {filename}: {type(e).__name__}") from e
