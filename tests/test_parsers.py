import json

import pytest

from app.ingestion.parsers import ParseError, parse_file


def make_pdf(text: str) -> bytes:
    """Smallest valid one-page PDF containing `text` (Helvetica)."""
    stream = f"BT /F1 12 Tf 50 700 Td ({text}) Tj ET".encode()
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out, offsets = bytearray(b"%PDF-1.4\n"), []
    for i, o in enumerate(objs, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + o + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    for off in offsets:
        out += b"%010d 00000 n \n" % off
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF" % (len(objs) + 1, xref)
    return bytes(out)


def test_txt_short_first_line_is_the_title():
    d = parse_file("incident.txt", b"Incident 42\r\n\r\nPayments failed.\r\n\r\n\r\n\r\nRolled back.")
    assert d.title == "Incident 42" and d.doc_type == "txt"
    assert d.text == "Payments failed.\n\nRolled back."


def test_txt_single_line_keeps_filename_title():
    d = parse_file("note.txt", b"Payments failed.")
    assert d.title == "note" and d.text == "Payments failed."


def test_md_title_from_heading():
    d = parse_file("x.md", b"# Outage Report\n\nSomething broke.")
    assert d.title == "Outage Report" and d.doc_type == "md"


def test_html_strips_scripts_and_reads_title():
    html = b"<html><head><title>Change Log</title><script>var x=1;</script></head><body><p>Deploy A</p><p>Deploy B</p></body></html>"
    d = parse_file("log.html", html)
    assert d.title == "Change Log"
    assert "var x" not in d.text and "Deploy A" in d.text and "Deploy B" in d.text


def test_json_text_and_metadata():
    raw = json.dumps({"title": "Svc", "text": "Gateway timed out", "owner": "ops"}).encode()
    d = parse_file("svc.json", raw)
    assert d.title == "Svc" and d.text == "Gateway timed out"
    assert d.metadata["owner"] == "ops" and d.metadata["filename"] == "svc.json"


def test_pdf_text_and_pages():
    d = parse_file("report.pdf", make_pdf("Payment outage in March"))
    assert "Payment outage in March" in d.text
    assert d.metadata["pages"] == 1 and d.doc_type == "pdf"


@pytest.mark.parametrize("name,data", [
    ("a.exe", b"x"),
    ("noext", b"x"),
    ("bad.json", b"{not json"),
    ("nofield.json", b'{"a": 1}'),
    ("broken.pdf", b"this is not a pdf"),
])
def test_bad_input_raises_parse_error(name, data):
    with pytest.raises(ParseError):
        parse_file(name, data)
