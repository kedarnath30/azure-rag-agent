from pathlib import Path

import pytest

from ragagent.documents import Document, chunk_document, load_documents, split_sections

ROOT = Path(__file__).resolve().parents[1]


def test_load_documents_reads_all_docs_with_titles() -> None:
    docs = load_documents(ROOT / "data" / "docs")
    assert len(docs) == 15
    pto = next(d for d in docs if d.doc_id == "pto-policy")
    assert pto.title == "Paid Time Off (PTO) Policy"


def test_load_documents_empty_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_documents(tmp_path)


def test_split_sections_uses_headings() -> None:
    text = "# Title\n\nIntro line.\n\n## First\nAlpha.\n\n## Second\nBeta."
    assert split_sections(text) == [
        ("Overview", "Intro line."),
        ("First", "Alpha."),
        ("Second", "Beta."),
    ]


def test_chunking_windows_overlap_and_ids_are_key_safe() -> None:
    body = " ".join(f"w{i}" for i in range(25))
    doc = Document("demo-doc", "Demo", f"# Demo\n## Part\n{body}")
    chunks = chunk_document(doc, chunk_size=10, overlap=2)
    assert [c.chunk_id for c in chunks] == ["demo-doc-0", "demo-doc-1", "demo-doc-2"]
    assert chunks[0].text.split()[-2:] == chunks[1].text.split()[:2]
    assert chunks[-1].text.split()[-1] == "w24"
    assert all(c.section == "Part" for c in chunks)


def test_chunking_rejects_bad_overlap() -> None:
    with pytest.raises(ValueError):
        chunk_document(Document("d", "D", "text"), chunk_size=5, overlap=5)
