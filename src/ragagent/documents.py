"""Load markdown documents and split them into overlapping chunks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Document:
    doc_id: str  # file name without extension, e.g. "pto-policy"
    title: str
    text: str


@dataclass(frozen=True)
class Chunk:
    chunk_id: str  # "<doc_id>-<n>"; only letters, digits and dashes (Azure AI Search key rules)
    doc_id: str
    title: str
    section: str
    text: str

    @property
    def search_text(self) -> str:
        """Text that gets embedded: title and section add useful context words."""
        return f"{self.title}. {self.section}. {self.text}"


def load_documents(docs_dir: Path) -> list[Document]:
    paths = sorted(Path(docs_dir).glob("*.md"))
    if not paths:
        raise FileNotFoundError(f"No .md files found in {docs_dir}")
    docs = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        title = next(
            (line[2:].strip() for line in text.splitlines() if line.startswith("# ")),
            path.stem,
        )
        docs.append(Document(doc_id=path.stem, title=title, text=text))
    return docs


def split_sections(text: str) -> list[tuple[str, str]]:
    """Split markdown into (heading, body) pairs on '## ' headings.

    Text before the first '## ' heading goes into an "Overview" section.
    The '# ' title line is skipped because the title is stored separately.
    """
    sections: list[tuple[str, str]] = []
    heading, lines = "Overview", []
    for line in text.splitlines():
        if line.startswith("## "):
            if any(s.strip() for s in lines):
                sections.append((heading, " ".join(s.strip() for s in lines if s.strip())))
            heading, lines = line[3:].strip(), []
        elif not line.startswith("# "):
            lines.append(line)
    if any(s.strip() for s in lines):
        sections.append((heading, " ".join(s.strip() for s in lines if s.strip())))
    return sections


def chunk_document(doc: Document, chunk_size: int = 120, overlap: int = 20) -> list[Chunk]:
    """Chunk each section into windows of `chunk_size` words that share `overlap` words."""
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")
    step = chunk_size - overlap
    chunks: list[Chunk] = []
    for section, body in split_sections(doc.text):
        words = body.split()
        for start in range(0, len(words), step):
            window = words[start : start + chunk_size]
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}-{len(chunks)}",
                    doc_id=doc.doc_id,
                    title=doc.title,
                    section=section,
                    text=" ".join(window),
                )
            )
            if start + chunk_size >= len(words):
                break
    return chunks


def chunk_documents(docs: list[Document], chunk_size: int = 120, overlap: int = 20) -> list[Chunk]:
    return [chunk for doc in docs for chunk in chunk_document(doc, chunk_size, overlap)]
