"""Document processing: validation of bad uploads, page-accurate extraction, chunk metadata."""
import pytest

from app.documents import InvalidDocumentError, chunk_document, clean_text, compute_doc_id, parse_pdf
from tests.conftest import PHYSICS_PAGES, make_pdf

MB = 1024 * 1024


def parse(name, data, max_bytes=25 * MB, max_pages=500):
    return parse_pdf(name, data, max_bytes=max_bytes, max_pages=max_pages)


def test_valid_pdf_keeps_page_numbers(physics_pdf):
    doc = parse("physics.pdf", physics_pdf)
    assert doc.num_pages == 3
    assert [p.page for p in doc.pages] == [1, 2, 3]
    assert "boils at 100 degrees" in doc.pages[0].text
    assert "Photosynthesis" in doc.pages[1].text


def test_chunks_carry_document_and_page_metadata(physics_pdf):
    doc = parse("physics.pdf", physics_pdf)
    chunks, ids = chunk_document(doc, chunk_size=500, chunk_overlap=50)
    assert len(chunks) == len(ids) == 3
    assert len(set(ids)) == len(ids)
    for c in chunks:
        assert c.metadata["source"] == "physics.pdf"
        assert c.metadata["doc_id"] == doc.doc_id
    assert [c.metadata["page"] for c in chunks] == [1, 2, 3]


def test_long_page_is_split_into_overlapping_chunks():
    long_text = " ".join(f"Sentence number {i} talks about topic {i % 7}." for i in range(80))  # fits on one page
    doc = parse("long.pdf", make_pdf([long_text]))
    assert doc.num_pages == 1
    chunks, _ = chunk_document(doc, chunk_size=60, chunk_overlap=10)
    assert len(chunks) > 5
    assert all(c.metadata["page"] == 1 for c in chunks)


def test_doc_id_is_content_hash(physics_pdf):
    assert compute_doc_id(physics_pdf) == compute_doc_id(bytes(physics_pdf))
    assert compute_doc_id(physics_pdf) != compute_doc_id(make_pdf(["something else entirely"]))


@pytest.mark.parametrize(
    "name,data,code,fragment",
    [
        ("notes.txt", b"just text", 415, "Only PDF"),
        ("empty.pdf", b"", 400, "empty"),
        ("fake.pdf", b"This is a text file renamed to .pdf", 415, "not a valid PDF"),
        ("broken.pdf", b"%PDF-1.7\n" + b"\x00garbage" * 50, 400, "corrupted"),
    ],
)
def test_invalid_files_are_rejected_with_clear_errors(name, data, code, fragment):
    with pytest.raises(InvalidDocumentError) as exc:
        parse(name, data)
    assert exc.value.status_code == code
    assert fragment.lower() in str(exc.value).lower()


def test_too_large_file_rejected(physics_pdf):
    with pytest.raises(InvalidDocumentError) as exc:
        parse("big.pdf", physics_pdf, max_bytes=100)
    assert exc.value.status_code == 413


def test_too_many_pages_rejected(physics_pdf):
    with pytest.raises(InvalidDocumentError) as exc:
        parse("many.pdf", physics_pdf, max_pages=2)
    assert exc.value.status_code == 413


def test_scanned_or_blank_pdf_rejected():
    with pytest.raises(InvalidDocumentError) as exc:
        parse("scan.pdf", make_pdf(["", ""]))
    assert "no extractable text" in str(exc.value).lower()


def test_password_protected_pdf_rejected():
    with pytest.raises(InvalidDocumentError) as exc:
        parse("secret.pdf", make_pdf(PHYSICS_PAGES, password="hunter2"))
    assert "password" in str(exc.value).lower()


def test_clean_text_fixes_hyphenation_and_control_chars():
    assert clean_text("retrie-\nval aug\x0cmented   text") == "retrieval augmented text"
