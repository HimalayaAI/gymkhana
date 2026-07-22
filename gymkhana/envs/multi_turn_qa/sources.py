"""Dataset and PDF source adapters for multi-turn QA generation."""

from __future__ import annotations

import hashlib
import io
import json
import logging
import re
from itertools import islice
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Optional

from .models import MultiTurnQAConfig, SourceDocument
from .profiles import DomainProfile


logger = logging.getLogger(__name__)


def json_safe(value: Any) -> Any:
    """Convert arbitrary source metadata into JSON-serializable values."""

    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        if isinstance(value, Mapping):
            return {str(key): json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_safe(item) for item in value]
        return str(value)


def stable_id(*parts: str) -> str:
    """Build a stable short identifier from source identity fields."""

    payload = "\n".join(parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:24]


def _iter_local_rows(path: Path) -> Iterator[Mapping[str, Any]]:
    if path.suffix.casefold() in {".jsonl", ".ndjson"}:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"local JSONL row {line_number} is not an object")
                yield value
        return
    if path.suffix.casefold() == ".json":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        rows = payload if isinstance(payload, list) else payload.get("rows", [])
        if not isinstance(rows, list):
            raise ValueError("local JSON must be a list or contain a rows list")
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("local JSON rows must be objects")
            yield row
        return
    raise ValueError(f"unsupported local dataset format: {path}")


def _iter_huggingface_rows(
    dataset_name: str,
    dataset_config: str,
    split: str,
    offset: int,
) -> Iterator[Mapping[str, Any]]:
    import requests

    cursor = offset
    page_size = 100
    while True:
        response = requests.get(
            "https://datasets-server.huggingface.co/rows",
            params={
                "dataset": dataset_name,
                "config": dataset_config,
                "split": split,
                "offset": cursor,
                "length": page_size,
            },
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        rows = payload.get("rows") or []
        if not rows:
            return
        for entry in rows:
            row = entry.get("row") if isinstance(entry, Mapping) else None
            if not isinstance(row, dict):
                raise ValueError("Hugging Face rows API returned an invalid row")
            yield row
        cursor += len(rows)
        total = payload.get("num_rows_total")
        if (isinstance(total, int) and cursor >= total) or len(rows) < page_size:
            return


class SourceLoader:
    """Normalize local, Hugging Face text, and Hugging Face PDF datasets."""

    def __init__(
        self,
        config: MultiTurnQAConfig,
        records: Optional[Iterable[Mapping[str, Any]]] = None,
    ) -> None:
        self.config = config
        self.records = [dict(record) for record in records] if records is not None else None

    def _dataset_rows(self) -> Iterable[Mapping[str, Any]]:
        settings = self.config.dataset
        offset = settings.dataset_offset
        if self.records is not None:
            return islice(self.records, offset, None)
        if not settings.dataset_name:
            return iter(())

        local_path = Path(settings.dataset_name).expanduser()
        if settings.dataset_backend in {"auto", "local"} and local_path.exists():
            return islice(_iter_local_rows(local_path), offset, None)
        if settings.dataset_backend == "local":
            raise FileNotFoundError(f"dataset file does not exist: {local_path}")
        if settings.dataset_backend == "huggingface-rows":
            return _iter_huggingface_rows(
                settings.dataset_name,
                settings.dataset_config or "default",
                settings.dataset_split,
                offset,
            )
        if settings.dataset_backend not in {"auto", "huggingface"}:
            raise ValueError(f"unsupported dataset backend: {settings.dataset_backend}")

        from datasets import Pdf, load_dataset

        kwargs: dict[str, Any] = {"split": settings.dataset_split, "streaming": True}
        dataset = (
            load_dataset(settings.dataset_name, settings.dataset_config, **kwargs)
            if settings.dataset_config
            else load_dataset(settings.dataset_name, **kwargs)
        )
        if "pdf" in getattr(dataset, "column_names", []):
            dataset = dataset.cast_column("pdf", Pdf(decode=False))
        return islice(dataset, offset, None)

    def _mapped_value(self, row: Mapping[str, Any], field: str) -> Any:
        source = self.config.dataset.field_mapping.get(field)
        return row.get(source) if source else None

    @staticmethod
    def _is_frontmatter(title: str) -> bool:
        lowered = title.casefold()
        return bool(
            "frontmatter" in lowered
            or "front matter" in lowered
            or "preface" in lowered
            or re.search(r"\bchapter\s*0\b", lowered)
        )

    def _text_document(self, row: Mapping[str, Any], row_index: int) -> SourceDocument:
        text = str(self._mapped_value(row, "text") or "").strip()
        source = str(
            self._mapped_value(row, "source")
            or self.config.dataset.dataset_name
            or "in-memory"
        )
        title_value = self._mapped_value(row, "title")
        title = str(title_value).strip() if title_value is not None else None
        identifier = self._mapped_value(row, "id")
        doc_id = str(identifier) if identifier is not None else stable_id(source, text)
        mapped_fields = {
            source_field
            for source_field in self.config.dataset.field_mapping.values()
            if source_field
        }
        metadata = {
            key: json_safe(value)
            for key, value in row.items()
            if key not in mapped_fields
        }
        metadata["dataset_name"] = self.config.dataset.dataset_name or "in-memory"
        metadata["dataset_split"] = self.config.dataset.dataset_split
        metadata["row_index"] = row_index
        return SourceDocument(
            id=doc_id,
            text=text,
            source=source,
            title=title,
            subject=(
                str(self._mapped_value(row, "subject"))
                if self._mapped_value(row, "subject") is not None
                else None
            ),
            grade=(
                str(self._mapped_value(row, "grade"))
                if self._mapped_value(row, "grade") is not None
                else None
            ),
            language=(
                str(self._mapped_value(row, "language"))
                if self._mapped_value(row, "language") is not None
                else self.config.generation.source_language
            ),
            license=(
                str(self._mapped_value(row, "license"))
                if self._mapped_value(row, "license") is not None
                else self.config.generation.source_license
            ),
            jurisdiction=(
                str(self._mapped_value(row, "jurisdiction"))
                if self._mapped_value(row, "jurisdiction") is not None
                else None
            ),
            document_date=(
                str(self._mapped_value(row, "document_date"))
                if self._mapped_value(row, "document_date") is not None
                else None
            ),
            metadata=metadata,
        )

    @staticmethod
    def _pdf_title(pdf_value: Any, row_index: int) -> str:
        if isinstance(pdf_value, Mapping):
            path = pdf_value.get("path") or pdf_value.get("src")
            if path:
                return Path(str(path).split("?")[0]).name
        return f"document-{row_index}.pdf"

    @staticmethod
    def _extract_pdf_pages(pdf_value: Any) -> list[tuple[int, str]]:
        from pypdf import PdfReader

        if not isinstance(pdf_value, Mapping):
            raise ValueError("PDF dataset row must contain a path/bytes mapping")
        raw_bytes = pdf_value.get("bytes")
        path = pdf_value.get("path") or pdf_value.get("src")
        if raw_bytes is not None:
            handle: Any = io.BytesIO(raw_bytes)
        elif not path:
            raise ValueError("PDF row has neither bytes nor path")
        elif str(path).startswith(("http://", "https://")):
            import requests

            response = requests.get(str(path), timeout=120)
            response.raise_for_status()
            handle = io.BytesIO(response.content)
        elif Path(str(path)).expanduser().exists():
            handle = Path(str(path)).expanduser().open("rb")
        else:
            from huggingface_hub import HfFileSystem

            remote_path = str(path)
            if remote_path.startswith("hf://"):
                remote_path = remote_path[len("hf://") :]
            handle = HfFileSystem().open(remote_path, "rb")

        try:
            reader = PdfReader(handle)
            pages: list[tuple[int, str]] = []
            for page_number, page in enumerate(reader.pages, start=1):
                text = (page.extract_text() or "").strip()
                if text:
                    pages.append((page_number, text))
            return pages
        finally:
            handle.close()

    @staticmethod
    def _chunk_pages(
        pages: list[tuple[int, str]],
        chunk_size: int,
        overlap: int,
    ) -> list[tuple[int, int, str]]:
        if not pages:
            return []
        parts: list[str] = []
        spans: list[tuple[int, int, int]] = []
        cursor = 0
        for page_number, text in pages:
            if parts:
                parts.append("\n\n")
                cursor += 2
            start = cursor
            parts.append(text)
            cursor += len(text)
            spans.append((start, cursor, page_number))
        combined = "".join(parts)
        chunks: list[tuple[int, int, str]] = []
        start = 0
        while start < len(combined):
            end = min(start + chunk_size, len(combined))
            if end < len(combined):
                boundary = combined.rfind(" ", start + chunk_size // 2, end)
                if boundary > start:
                    end = boundary
            text = combined[start:end].strip()
            overlapping_pages = [
                page
                for page_start, page_end, page in spans
                if page_end > start and page_start < end
            ]
            if text and overlapping_pages:
                chunks.append((min(overlapping_pages), max(overlapping_pages), text))
            if end >= len(combined):
                break
            next_start = max(start + 1, end - overlap)
            while next_start < len(combined) and combined[next_start].isspace():
                next_start += 1
            start = next_start
        return chunks

    def _pdf_documents(
        self,
        row: Mapping[str, Any],
        row_index: int,
        profile: DomainProfile,
    ) -> list[SourceDocument]:
        pdf_value = self._mapped_value(row, "pdf") or row.get("pdf")
        title = self._pdf_title(pdf_value, row_index)
        pages = self._extract_pdf_pages(pdf_value)
        settings = self.config.generation
        chunks = self._chunk_pages(
            pages,
            chunk_size=settings.chunk_size_chars,
            overlap=settings.chunk_overlap_chars,
        )[: settings.max_chunks_per_document]
        source_name = self.config.dataset.dataset_name or "PDF dataset"
        documents: list[SourceDocument] = []
        for chunk_index, (page_start, page_end, text) in enumerate(chunks):
            documents.append(
                SourceDocument(
                    id=f"{stable_id(source_name, title)}-chunk-{chunk_index:04d}",
                    text=text,
                    source=source_name,
                    title=title,
                    language=self.config.generation.source_language or "ne",
                    license=self.config.generation.source_license,
                    jurisdiction=profile.jurisdiction,
                    page_start=page_start,
                    page_end=page_end,
                    metadata={
                        "dataset_name": source_name,
                        "dataset_split": self.config.dataset.dataset_split,
                        "row_index": row_index,
                        "pdf_title": title,
                        "chunk_index": chunk_index,
                    },
                )
            )
        return documents

    def load(self, limit: Optional[int], profile: DomainProfile) -> list[SourceDocument]:
        """Load and normalize at most ``limit`` source chunks."""

        settings = self.config.generation
        if limit is not None and limit < 0:
            raise ValueError("limit must be non-negative")
        if limit == 0:
            return []
        subjects = {value.casefold() for value in settings.subjects}
        grades = {value.casefold() for value in settings.grades}
        documents_out: list[SourceDocument] = []
        for row_index, row in enumerate(
            self._dataset_rows(), start=self.config.dataset.dataset_offset
        ):
            pdf_value = self._mapped_value(row, "pdf") or row.get("pdf")
            source_kind = settings.source_kind
            is_pdf = source_kind == "pdf" or (
                source_kind == "auto" and pdf_value is not None
            )
            try:
                documents = (
                    self._pdf_documents(row, row_index, profile)
                    if is_pdf
                    else [self._text_document(row, row_index)]
                )
            except Exception as error:
                logger.warning(
                    "Skipping source row %s after extraction failure: %s",
                    row_index,
                    error,
                )
                continue
            for document in documents:
                if document.jurisdiction is None and profile.jurisdiction:
                    document.jurisdiction = profile.jurisdiction
                if len(document.text) < settings.min_source_chars:
                    logger.info("Skipping short source %s", document.id)
                    continue
                if (
                    settings.skip_frontmatter
                    and document.title
                    and self._is_frontmatter(document.title)
                ):
                    logger.info("Skipping front matter %s", document.id)
                    continue
                if subjects and (
                    not document.subject or document.subject.casefold() not in subjects
                ):
                    continue
                if grades and (
                    not document.grade or document.grade.casefold() not in grades
                ):
                    continue
                documents_out.append(document)
                if limit is not None and len(documents_out) >= limit:
                    return documents_out
        return documents_out


__all__ = ["SourceLoader", "json_safe", "stable_id"]
