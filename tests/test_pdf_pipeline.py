import asyncio
import base64
import json

import pytest
from fastapi.testclient import TestClient

from modules.multimodal.app import create_app
from modules.multimodal.config import Settings
from modules.multimodal.media import find_media_items
from modules.multimodal.pdf import (
    HttpPdfOcrProvider,
    PyMuPdfExtractor,
    PyMuPdfRenderer,
    QwenVisionPdfOcrProvider,
    WorkPdfExtractProvider,
    PdfExtractError,
    PdfLimitError,
    PdfQueueFullError,
    PdfOcrPage,
    PdfPage,
    PdfProcessResult,
    build_pdf_evidence,
    choose_ocr_pages,
    enforce_pdf_limits,
    prepare_pdf_media,
    PdfWorkLimiter,
    decode_pdf_block,
    process_pdf_documents,
)


def _pdf_block(data: bytes) -> dict:
    return {
        "type": "file",
        "file": {
            "filename": "report.pdf",
            "file_data": "data:application/pdf;base64," + base64.b64encode(data).decode(),
        },
    }


def test_detects_pdf_as_document_media():
    items = find_media_items({"messages": [{"role": "user", "content": [_pdf_block(b"%PDF-1.7")]}]})
    assert len(items) == 1
    assert items[0].kind == "document"
    assert items[0].mime_type == "application/pdf"
    assert items[0].filename == "report.pdf"


def test_pdf_is_rejected_before_upstream_when_pdf_pipeline_is_disabled():
    app = create_app(Settings(bridge_token="test", pdf_enabled=False))
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer test"},
            json={
                "model": "deepseek-v4-mm-bridge",
                "messages": [
                    {"role": "user", "content": [_pdf_block(b"%PDF-disabled")]}
                ],
            },
        )
    assert response.status_code == 501
    assert response.json()["error"]["stage"] == "pdf_preprocessing_disabled"


def test_enforces_pdf_size_and_page_and_pixel_limits_before_ocr():
    settings = Settings(pdf_max_bytes=8, pdf_max_pages=2, pdf_max_rendered_pixels=1_000)
    with pytest.raises(PdfLimitError, match="file size"):
        enforce_pdf_limits(pdf_bytes=b"123456789", page_count=1, rendered_pixels=0, settings=settings)
    with pytest.raises(PdfLimitError, match="page count"):
        enforce_pdf_limits(pdf_bytes=b"1234", page_count=3, rendered_pixels=0, settings=settings)
    with pytest.raises(PdfLimitError, match="rendered pixels"):
        enforce_pdf_limits(pdf_bytes=b"1234", page_count=2, rendered_pixels=1_001, settings=settings)


def test_only_pages_below_native_text_threshold_are_selected_for_ocr():
    pages = [
        PdfPage(number=1, native_text="enough native text here"),
        PdfPage(number=2, native_text="x"),
        PdfPage(number=3, native_text="   "),
    ]
    assert choose_ocr_pages(pages, min_native_text_chars=5) == [2, 3]


def test_pdf_evidence_preserves_page_numbers_and_separates_text_from_interpretation():
    result = PdfProcessResult(
        filename="report.pdf",
        pages=[
            PdfPage(number=1, native_text="Native page one"),
            PdfPage(number=2, native_text=""),
        ],
        ocr_pages=[
            PdfOcrPage(number=2, ocr_text="OCR page two", interpretation="A chart rises."),
        ],
    )
    evidence = json.loads(build_pdf_evidence(result))
    assert evidence["document"]["filename"] == "report.pdf"
    assert evidence["document"]["pages"] == [
        {"page_number": 1, "native_text": "Native page one", "ocr_text": "", "interpretation": "", "source": ""},
        {"page_number": 2, "native_text": "", "ocr_text": "OCR page two", "interpretation": "A chart rises.", "source": ""},
    ]


def test_qwen_provider_receives_rendered_pages_not_raw_pdf_bytes():
    calls = []

    class FakeExtractor:
        def extract(self, pdf_bytes, settings):
            assert pdf_bytes == b"%PDF-test"
            return [PdfPage(number=1, native_text=""), PdfPage(number=2, native_text="native enough")]

    class FakeRenderer:
        def render(self, pdf_bytes, page_numbers, settings):
            assert pdf_bytes == b"%PDF-test"
            assert page_numbers == [1]
            return {1: (b"PNG-PAGE-1", 100, 100)}

    class FakeProvider:
        def ocr(self, *, page_number, image_bytes, mime_type, timeout_seconds):
            calls.append((page_number, image_bytes, mime_type, timeout_seconds))
            assert b"%PDF" not in image_bytes
            return PdfOcrPage(number=page_number, ocr_text="read me", interpretation="heading")

    settings = Settings(
        pdf_enabled=True,
        pdf_max_bytes=1_000,
        pdf_max_pages=5,
        pdf_max_rendered_pixels=20_000,
        pdf_min_native_text_chars=5,
        pdf_ocr_timeout_seconds=7,
    )
    result = prepare_pdf_media(
        pdf_bytes=b"%PDF-test",
        filename="x.pdf",
        settings=settings,
        extractor=FakeExtractor(),
        renderer=FakeRenderer(),
        provider=FakeProvider(),
    )
    assert calls == [(1, b"PNG-PAGE-1", "image/png", 7)]
    assert result.pages[1].native_text == "native enough"


def test_decode_pdf_block_returns_bytes_and_filename():
    item = find_media_items(_pdf_block(b"%PDF-data"))[0]
    raw, filename = decode_pdf_block(item)
    assert raw == b"%PDF-data"
    assert filename == "report.pdf"


def test_http_provider_payload_is_page_image_only():
    captured = {}

    def transport(request):
        captured["json"] = json.loads(request.content)
        return __import__("httpx").Response(
            200,
            json={"page_number": 3, "ocr_text": "read", "interpretation": "table"},
        )

    import httpx

    provider = HttpPdfOcrProvider(
        url="http://ocr.internal/ocr",
        api_key="secret",
        client=httpx.Client(transport=httpx.MockTransport(transport)),
    )
    result = provider.ocr(
        page_number=3,
        image_bytes=b"PNG-DATA",
        mime_type="image/png",
        timeout_seconds=5,
    )
    assert result.ocr_text == "read"
    assert captured["json"] == {
        "page_number": 3,
        "mime_type": "image/png",
        "image_base64": base64.b64encode(b"PNG-DATA").decode(),
    }
    assert "%PDF" not in json.dumps(captured["json"])


def test_qwen_provider_sends_page_png_not_pdf_and_separates_fields():
    captured = {}

    def transport(request):
        captured["json"] = json.loads(request.content)
        return __import__("httpx").Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "ocr_text": "TOTAL 100",
                                    "interpretation": "invoice total",
                                }
                            )
                        }
                    }
                ]
            },
        )

    import httpx

    provider = QwenVisionPdfOcrProvider(
        root_url="http://qwen.internal:8083",
        api_key="secret",
        model="qwen-vision",
        max_tokens=512,
        client=httpx.Client(transport=httpx.MockTransport(transport)),
    )
    result = provider.ocr(
        page_number=2,
        image_bytes=b"PNG-PAGE",
        mime_type="image/png",
        timeout_seconds=5,
    )
    payload = captured["json"]
    image_url = payload["messages"][1]["content"][1]["image_url"]["url"]
    assert image_url.startswith("data:image/png;base64,")
    assert "%PDF" not in json.dumps(payload)
    assert result == PdfOcrPage(
        number=2, ocr_text="TOTAL 100", interpretation="invoice total"
    )


def test_pymupdf_extracts_native_text_and_renders_only_requested_pages():
    import fitz

    document = fitz.open()
    page1 = document.new_page(width=200, height=100)
    page1.insert_text((20, 40), "Native page text")
    document.new_page(width=200, height=100)
    raw = document.tobytes()

    settings = Settings(pdf_render_dpi=72)
    pages = PyMuPdfExtractor().extract(raw, settings)
    assert [page.number for page in pages] == [1, 2]
    assert "Native page text" in pages[0].native_text
    assert pages[1].native_text == ""

    rendered = PyMuPdfRenderer().render(raw, [2], settings)
    assert list(rendered) == [2]
    png, width, height = rendered[2]
    assert png.startswith(b"\x89PNG")
    assert (width, height) == (200, 100)


def test_pdf_provider_url_is_optional_until_manus_supplies_endpoint(monkeypatch):
    monkeypatch.delenv("MM_PDF_OCR_URL", raising=False)
    settings = Settings.from_env()
    assert settings.pdf_ocr_provider == "disabled"
    assert settings.pdf_ocr_url == ""


def test_llama_api_key_can_be_loaded_from_secret_file(tmp_path, monkeypatch):
    secret = tmp_path / "ray-key"
    secret.write_text("secret-from-file\n")
    monkeypatch.setenv("LLAMA_API_KEY", "wrong-env-value")
    monkeypatch.setenv("LLAMA_API_KEY_FILE", str(secret))
    settings = Settings.from_env()
    assert settings.llama_api_key == "secret-from-file"


def test_work_limiter_rejects_above_bounded_queue():
    async def scenario():
        limiter = PdfWorkLimiter(max_concurrency=1, max_queue=1)
        entered = asyncio.Event()
        release = asyncio.Event()

        async def active():
            async with limiter.slot():
                entered.set()
                await release.wait()

        async def queued():
            async with limiter.slot():
                return "queued-ran"

        active_task = asyncio.create_task(active())
        await entered.wait()
        queued_task = asyncio.create_task(queued())
        await asyncio.sleep(0)

        with pytest.raises(PdfQueueFullError):
            async with limiter.slot():
                pass

        release.set()
        await active_task
        assert await queued_task == "queued-ran"

    asyncio.run(scenario())


def test_process_pdf_documents_returns_page_numbered_evidence():
    item = find_media_items(_pdf_block(b"%PDF-data"))[0]

    class Extractor:
        def extract(self, pdf_bytes, settings):
            return [PdfPage(number=1, native_text="native")]

    class Renderer:
        def render(self, pdf_bytes, page_numbers, settings):
            return {}

    class Provider:
        def ocr(self, **kwargs):
            raise AssertionError("OCR must not run for native-text page")

    settings = Settings(
        pdf_enabled=True,
        pdf_min_native_text_chars=1,
        pdf_timeout_seconds=2,
    )
    evidence = asyncio.run(
        process_pdf_documents(
            [item],
            settings=settings,
            extractor=Extractor(),
            renderer=Renderer(),
            provider=Provider(),
            limiter=PdfWorkLimiter(max_concurrency=1, max_queue=1),
        )
    )
    payload = json.loads(evidence)
    assert payload["documents"][0]["pages"][0] == {
        "page_number": 1,
        "native_text": "native",
        "ocr_text": "",
        "interpretation": "",
        "source": "",
    }


def test_work_pdf_extract_provider_posts_raw_pdf_bytes_not_images():
    captured = {}

    def transport(request):
        captured["url"] = str(request.url)
        captured["content_type"] = request.headers["content-type"]
        captured["body"] = request.content
        return httpx.Response(
            200,
            json={
                "status": "ok",
                "pages": [
                    {
                        "page": 1,
                        "source": "native_text",
                        "native_text": "特許翻訳",
                        "ocr_text": "",
                        "interpretation": None,
                        "rendered_pixels": 0,
                    },
                    {
                        "page": 2,
                        "source": "tesseract_ocr",
                        "native_text": "",
                        "ocr_text": "Patent Translation OCR TEST",
                        "interpretation": None,
                        "rendered_pixels": 3825000,
                    },
                ],
                "native_text": "特許翻訳",
                "ocr_text": "Patent Translation OCR TEST",
                "interpretation": None,
            },
        )

    import httpx

    provider = WorkPdfExtractProvider(
        url="http://ocr.internal/v1/extract",
        client=httpx.Client(transport=httpx.MockTransport(transport)),
    )
    result = provider.extract(
        pdf_bytes=b"%PDF-data",
        filename="report.pdf",
        timeout_seconds=5,
    )
    assert captured["url"] == "http://ocr.internal/v1/extract"
    assert captured["content_type"] == "application/pdf"
    assert captured["body"] == b"%PDF-data"
    evidence = json.loads(build_pdf_evidence(result))["document"]["pages"]
    assert evidence[0] == {
        "page_number": 1,
        "native_text": "特許翻訳",
        "ocr_text": "",
        "interpretation": "",
        "source": "native_text",
    }
    assert evidence[1]["ocr_text"] == "Patent Translation OCR TEST"
    assert evidence[1]["interpretation"] == ""
    assert evidence[1]["source"] == "tesseract_ocr"


def test_work_pdf_extract_provider_maps_backpressure_errors():
    import httpx

    def transport(request):
        return httpx.Response(429, json={"error": "queue_full"})

    provider = WorkPdfExtractProvider(
        url="http://ocr.internal/v1/extract",
        client=httpx.Client(transport=httpx.MockTransport(transport)),
    )
    with pytest.raises(PdfQueueFullError):
        provider.extract(pdf_bytes=b"%PDF-data", filename="x.pdf", timeout_seconds=5)


def test_work_pdf_extract_provider_maps_timeout_errors():
    import httpx

    def transport(request):
        return httpx.Response(504, json={"error": "processing_timeout"})

    provider = WorkPdfExtractProvider(
        url="http://ocr.internal/v1/extract",
        client=httpx.Client(transport=httpx.MockTransport(transport)),
    )
    with pytest.raises(PdfExtractError) as exc:
        provider.extract(pdf_bytes=b"%PDF-data", filename="x.pdf", timeout_seconds=5)
    assert exc.value.status_code == 504
