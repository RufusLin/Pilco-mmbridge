from __future__ import annotations

import asyncio
import base64
import json
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Protocol

import httpx

from .config import Settings
from .media import MediaItem


class PdfLimitError(ValueError):
    pass


class PdfQueueFullError(RuntimeError):
    pass


class PdfExtractError(RuntimeError):
    def __init__(self, message: str, *, status_code: int = 502, stage: str = "pdf_extract"):
        super().__init__(message)
        self.status_code = status_code
        self.stage = stage


@dataclass(frozen=True)
class PdfPage:
    number: int
    native_text: str
    source: str = ""


@dataclass(frozen=True)
class PdfOcrPage:
    number: int
    ocr_text: str
    interpretation: str = ""


@dataclass(frozen=True)
class PdfProcessResult:
    filename: str
    pages: list[PdfPage]
    ocr_pages: list[PdfOcrPage]


class PdfExtractor(Protocol):
    def extract(self, pdf_bytes: bytes, settings: Settings) -> list[PdfPage]: ...


class PdfRenderer(Protocol):
    def render(
        self,
        pdf_bytes: bytes,
        page_numbers: list[int],
        settings: Settings,
    ) -> dict[int, tuple[bytes, int, int]]: ...


class PdfOcrProvider(Protocol):
    def ocr(
        self,
        *,
        page_number: int,
        image_bytes: bytes,
        mime_type: str,
        timeout_seconds: float,
    ) -> PdfOcrPage: ...


class PyMuPdfExtractor:
    def extract(self, pdf_bytes: bytes, settings: Settings) -> list[PdfPage]:
        import fitz

        with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
            if document.page_count > settings.pdf_max_pages:
                raise PdfLimitError(
                    f"PDF page count {document.page_count} exceeds limit "
                    f"{settings.pdf_max_pages}"
                )
            return [
                PdfPage(number=index + 1, native_text=page.get_text("text").strip())
                for index, page in enumerate(document)
            ]


class PyMuPdfRenderer:
    def render(
        self,
        pdf_bytes: bytes,
        page_numbers: list[int],
        settings: Settings,
    ) -> dict[int, tuple[bytes, int, int]]:
        import fitz

        scale = settings.pdf_render_dpi / 72.0
        matrix = fitz.Matrix(scale, scale)
        rendered: dict[int, tuple[bytes, int, int]] = {}
        with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
            for page_number in page_numbers:
                if page_number < 1 or page_number > document.page_count:
                    raise PdfLimitError(f"invalid PDF page number {page_number}")
                pixmap = document.load_page(page_number - 1).get_pixmap(
                    matrix=matrix, alpha=False
                )
                page_pixels = pixmap.width * pixmap.height
                if page_pixels > settings.pdf_max_rendered_pixels:
                    raise PdfLimitError(
                        "PDF rendered pixels "
                        f"{page_pixels} exceeds per-page limit "
                        f"{settings.pdf_max_rendered_pixels}"
                    )
                rendered[page_number] = (
                    pixmap.tobytes("png"),
                    pixmap.width,
                    pixmap.height,
                )
        return rendered


def decode_pdf_block(item: MediaItem) -> tuple[bytes, str]:
    if item.kind != "document" or item.mime_type != "application/pdf":
        raise ValueError("media item is not a PDF document")
    block = item.block
    if not isinstance(block, dict):
        raise ValueError("PDF block must be an object")
    nested_file = block.get("file")
    file_obj = nested_file if isinstance(nested_file, dict) else block
    nested_source = block.get("source")
    source = nested_source if isinstance(nested_source, dict) else {}
    data = file_obj.get("file_data") or file_obj.get("data") or source.get("data")
    if not isinstance(data, str):
        raise ValueError("PDF block does not contain base64 data")
    if data.lower().startswith("data:application/pdf;base64,"):
        data = data.split(",", 1)[1]
    try:
        raw = base64.b64decode(data, validate=True)
    except Exception as exc:
        raise ValueError("PDF block contains invalid base64 data") from exc
    if not raw.startswith(b"%PDF-"):
        raise ValueError("decoded document is not a PDF")
    return raw, item.filename or "document.pdf"


class HttpPdfOcrProvider:
    """Backend-neutral OCR transport. The payload can never contain PDF bytes."""

    def __init__(
        self,
        *,
        url: str,
        api_key: str = "",
        client: httpx.Client | None = None,
    ):
        self.url = url
        self.api_key = api_key
        self.client = client or httpx.Client()

    def ocr(
        self,
        *,
        page_number: int,
        image_bytes: bytes,
        mime_type: str,
        timeout_seconds: float,
    ) -> PdfOcrPage:
        headers = (
            {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        )
        response = self.client.post(
            self.url,
            headers=headers,
            json={
                "page_number": page_number,
                "mime_type": mime_type,
                "image_base64": base64.b64encode(image_bytes).decode("ascii"),
            },
            timeout=timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        return PdfOcrPage(
            number=page_number,
            ocr_text=str(payload.get("ocr_text") or ""),
            interpretation=str(payload.get("interpretation") or ""),
        )


class WorkPdfExtractProvider:
    """Document-evidence client for the work-host /v1/extract service.

    The request body is raw PDF bytes. This provider never asks the OCR
    service to interpret images, and it never copies interpretation from OCR.
    """

    def __init__(
        self,
        *,
        url: str,
        client: httpx.Client | None = None,
    ):
        self.url = url
        self.client = client or httpx.Client()

    def extract(
        self,
        *,
        pdf_bytes: bytes,
        filename: str,
        timeout_seconds: float,
    ) -> PdfProcessResult:
        response = self.client.post(
            self.url,
            headers={"Content-Type": "application/pdf"},
            content=pdf_bytes,
            timeout=timeout_seconds,
        )
        if response.status_code == 429:
            raise PdfQueueFullError("PDF work queue is full")
        if response.status_code in {408, 504}:
            raise PdfExtractError(
                response.text or response.reason_phrase,
                status_code=response.status_code,
            )
        if response.status_code in {413, 415}:
            raise PdfLimitError(response.text or response.reason_phrase)
        if response.status_code >= 400:
            raise PdfExtractError(
                response.text or response.reason_phrase,
                status_code=502 if response.status_code >= 500 else response.status_code,
            )
        payload = response.json()
        pages: list[PdfPage] = []
        ocr_pages: list[PdfOcrPage] = []
        for entry in payload.get("pages") or []:
            number = int(entry.get("page") or entry.get("page_number") or 0)
            native_text = str(entry.get("native_text") or "")
            ocr_text = str(entry.get("ocr_text") or "")
            source = str(entry.get("source") or "")
            pages.append(PdfPage(number=number, native_text=native_text, source=source))
            if ocr_text:
                ocr_pages.append(
                    PdfOcrPage(number=number, ocr_text=ocr_text, interpretation="")
                )
        return PdfProcessResult(filename=filename, pages=pages, ocr_pages=ocr_pages)


class QwenVisionPdfOcrProvider:
    """OpenAI-compatible Qwen VLM provider for rendered PDF page images."""

    def __init__(
        self,
        *,
        root_url: str,
        api_key: str,
        model: str,
        max_tokens: int = 2048,
        client: httpx.Client | None = None,
    ):
        self.root_url = root_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.client = client or httpx.Client()

    def ocr(
        self,
        *,
        page_number: int,
        image_bytes: bytes,
        mime_type: str,
        timeout_seconds: float,
    ) -> PdfOcrPage:
        image_url = (
            f"data:{mime_type};base64,"
            + base64.b64encode(image_bytes).decode("ascii")
        )
        response = self.client.post(
            f"{self.root_url}/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Analyze one rendered PDF page. Return JSON only with "
                            "ocr_text containing verbatim visible text and "
                            "interpretation containing non-OCR visual structure or "
                            "meaning. Keep the two fields strictly separate."
                        ),
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"Rendered PDF page number: {page_number}",
                            },
                            {"type": "image_url", "image_url": {"url": image_url}},
                        ],
                    },
                ],
                "temperature": 0,
                "max_tokens": self.max_tokens,
                "response_format": {"type": "json_object"},
                "stream": False,
            },
            timeout=timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        text = payload["choices"][0]["message"]["content"]
        parsed = json.loads(text)
        return PdfOcrPage(
            number=page_number,
            ocr_text=str(parsed.get("ocr_text") or ""),
            interpretation=str(parsed.get("interpretation") or ""),
        )


class RayPdfInterpretationProvider:
    """Ray VLM interpretation for already-OCR'd sparse PDF pages."""

    def __init__(
        self,
        *,
        root_url: str,
        api_key: str,
        model: str,
        max_tokens: int = 1024,
        client: httpx.Client | None = None,
    ):
        self.root_url = root_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.client = client or httpx.Client()

    def interpret(
        self,
        *,
        page_number: int,
        image_bytes: bytes,
        mime_type: str,
        timeout_seconds: float,
    ) -> str:
        image_url = (
            f"data:{mime_type};base64,"
            + base64.b64encode(image_bytes).decode("ascii")
        )
        response = self.client.post(
            f"{self.root_url}/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Describe visual structure on one rendered PDF page. "
                            "Return JSON only with interpretation covering layout, "
                            "diagrams, tables, handwriting, and spatial relationships. "
                            "Do not transcribe text; OCR is supplied separately. "
                            "Do not answer the user."
                        ),
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"Rendered PDF page number: {page_number}",
                            },
                            {"type": "image_url", "image_url": {"url": image_url}},
                        ],
                    },
                ],
                "temperature": 0,
                "max_tokens": self.max_tokens,
                "response_format": {"type": "json_object"},
                "stream": False,
            },
            timeout=timeout_seconds,
        )
        if response.status_code >= 400:
            raise PdfExtractError(
                response.text or response.reason_phrase,
                status_code=502,
                stage="pdf_interpretation",
            )
        payload = response.json()
        text = payload["choices"][0]["message"]["content"]
        parsed = json.loads(text) if isinstance(text, str) else text
        return str((parsed or {}).get("interpretation") or "")


def enforce_pdf_limits(
    *,
    pdf_bytes: bytes,
    page_count: int,
    rendered_pixels: int,
    settings: Settings,
) -> None:
    if len(pdf_bytes) > settings.pdf_max_bytes:
        raise PdfLimitError(
            f"PDF file size {len(pdf_bytes)} exceeds limit {settings.pdf_max_bytes}"
        )
    if page_count > settings.pdf_max_pages:
        raise PdfLimitError(
            f"PDF page count {page_count} exceeds limit {settings.pdf_max_pages}"
        )
    if rendered_pixels > settings.pdf_max_rendered_pixels:
        raise PdfLimitError(
            "PDF rendered pixels "
            f"{rendered_pixels} exceeds limit {settings.pdf_max_rendered_pixels}"
        )


def choose_ocr_pages(
    pages: list[PdfPage], *, min_native_text_chars: int
) -> list[int]:
    return [
        page.number
        for page in pages
        if len(page.native_text.strip()) < min_native_text_chars
    ]


def prepare_pdf_media(
    *,
    pdf_bytes: bytes,
    filename: str,
    settings: Settings,
    extractor: PdfExtractor,
    renderer: PdfRenderer,
    provider: PdfOcrProvider,
) -> PdfProcessResult:
    # Reject oversized bytes before parsing, rendering, or external OCR.
    enforce_pdf_limits(
        pdf_bytes=pdf_bytes, page_count=0, rendered_pixels=0, settings=settings
    )
    pages = extractor.extract(pdf_bytes, settings)
    enforce_pdf_limits(
        pdf_bytes=pdf_bytes,
        page_count=len(pages),
        rendered_pixels=0,
        settings=settings,
    )
    page_numbers = choose_ocr_pages(
        pages, min_native_text_chars=settings.pdf_min_native_text_chars
    )
    rendered = renderer.render(pdf_bytes, page_numbers, settings) if page_numbers else {}
    rendered_pixels = sum(width * height for _, width, height in rendered.values())
    enforce_pdf_limits(
        pdf_bytes=pdf_bytes,
        page_count=len(pages),
        rendered_pixels=rendered_pixels,
        settings=settings,
    )

    ocr_pages: list[PdfOcrPage] = []
    for page_number in page_numbers:
        image_bytes, _, _ = rendered[page_number]
        # This provider boundary accepts images only. Raw PDF bytes cannot be
        # sent to Qwen, Tika/Tesseract, or future OCR backends.
        ocr_pages.append(
            provider.ocr(
                page_number=page_number,
                image_bytes=image_bytes,
                mime_type="image/png",
                timeout_seconds=settings.pdf_ocr_timeout_seconds,
            )
        )
    return PdfProcessResult(filename=filename, pages=pages, ocr_pages=ocr_pages)


def build_pdf_evidence(result: PdfProcessResult) -> str:
    ocr_by_page = {page.number: page for page in result.ocr_pages}
    pages = []
    for page in result.pages:
        ocr = ocr_by_page.get(page.number)
        pages.append(
            {
                "page_number": page.number,
                "native_text": page.native_text,
                "ocr_text": ocr.ocr_text if ocr else "",
                "interpretation": ocr.interpretation if ocr else "",
                "source": page.source,
            }
        )
    return json.dumps(
        {"document": {"filename": result.filename, "pages": pages}},
        ensure_ascii=False,
        separators=(",", ":"),
    )


class PdfWorkLimiter:
    def __init__(self, *, max_concurrency: int, max_queue: int):
        self._semaphore = asyncio.Semaphore(max(1, max_concurrency))
        self._max_queue = max(0, max_queue)
        self._waiting = 0
        self._state_lock = asyncio.Lock()

    @asynccontextmanager
    async def slot(self):
        queued = False
        async with self._state_lock:
            if self._semaphore.locked():
                if self._waiting >= self._max_queue:
                    raise PdfQueueFullError("PDF work queue is full")
                self._waiting += 1
                queued = True
        try:
            await self._semaphore.acquire()
        finally:
            if queued:
                async with self._state_lock:
                    self._waiting -= 1
        try:
            yield
        finally:
            self._semaphore.release()


async def process_pdf_documents(
    items: list[MediaItem],
    *,
    settings: Settings,
    extractor: PdfExtractor,
    renderer: PdfRenderer,
    provider: PdfOcrProvider,
    limiter: PdfWorkLimiter,
    extract_provider: WorkPdfExtractProvider | None = None,
    interpretation_provider: RayPdfInterpretationProvider | None = None,
) -> str:
    async with limiter.slot():
        results: list[dict] = []
        for item in items:
            raw, filename = decode_pdf_block(item)
            enforce_pdf_limits(
                pdf_bytes=raw, page_count=0, rendered_pixels=0, settings=settings
            )
            if extract_provider is not None:
                result = await asyncio.wait_for(
                    asyncio.to_thread(
                        _extract_and_interpret,
                        pdf_bytes=raw,
                        filename=filename,
                        settings=settings,
                        extract_provider=extract_provider,
                        renderer=renderer,
                        interpretation_provider=interpretation_provider,
                    ),
                    timeout=settings.pdf_timeout_seconds,
                )
            else:
                result = await asyncio.wait_for(
                    asyncio.to_thread(
                        prepare_pdf_media,
                        pdf_bytes=raw,
                        filename=filename,
                        settings=settings,
                        extractor=extractor,
                        renderer=renderer,
                        provider=provider,
                    ),
                    timeout=settings.pdf_timeout_seconds,
                )
            results.append(json.loads(build_pdf_evidence(result))["document"])
        return json.dumps(
            {"documents": results},
            ensure_ascii=False,
            separators=(",", ":"),
        )


def _extract_and_interpret(
    *,
    pdf_bytes: bytes,
    filename: str,
    settings: Settings,
    extract_provider: WorkPdfExtractProvider,
    renderer: PdfRenderer,
    interpretation_provider: RayPdfInterpretationProvider | None,
) -> PdfProcessResult:
    result = extract_provider.extract(
        pdf_bytes=pdf_bytes,
        filename=filename,
        timeout_seconds=settings.pdf_ocr_timeout_seconds,
    )
    sparse = [page.number for page in result.pages if page.source == "tesseract_ocr"]
    if not sparse or interpretation_provider is None:
        return result
    rendered = renderer.render(pdf_bytes, sparse, settings)
    ocr_by_page = {page.number: page for page in result.ocr_pages}
    merged: list[PdfOcrPage] = []
    seen: set[int] = set()
    for number in sparse:
        image_bytes, _, _ = rendered[number]
        interpretation = interpretation_provider.interpret(
            page_number=number,
            image_bytes=image_bytes,
            mime_type="image/png",
            timeout_seconds=settings.pdf_ocr_timeout_seconds,
        )
        existing = ocr_by_page.get(number)
        merged.append(
            PdfOcrPage(
                number=number,
                ocr_text=existing.ocr_text if existing else "",
                interpretation=interpretation,
            )
        )
        seen.add(number)
    for page in result.ocr_pages:
        if page.number not in seen:
            merged.append(page)
    return PdfProcessResult(
        filename=result.filename, pages=result.pages, ocr_pages=merged
    )
