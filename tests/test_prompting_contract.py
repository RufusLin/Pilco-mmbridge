from __future__ import annotations

import json
import unittest

from modules.multimodal.config import Settings
from modules.multimodal.media import MediaItem
from modules.multimodal.prompting import (
    ANALYZER_PROMPT_VERSION,
    ANALYZER_SYSTEM_PROMPT,
    build_analyzer_body,
    build_final_context,
)


class PromptingContractTests(unittest.TestCase):
    def test_output_shape_uses_only_evidence_blocks(self) -> None:
        shape_text = ANALYZER_SYSTEM_PROMPT.split("Output shape:\n", 1)[1]
        shape = json.loads(shape_text)

        self.assertEqual(list(shape), ["media"])
        media_item = shape["media"][0]
        self.assertEqual(media_item["type"], "image|audio|video|media")
        self.assertIn("text_blocks", media_item)
        self.assertNotIn("visual_blocks", media_item)
        self.assertNotIn("audio_blocks", media_item)
        self.assertIn("Include visual_blocks only when applicable.", ANALYZER_SYSTEM_PROMPT)
        self.assertIn("Include audio_blocks only when applicable.", ANALYZER_SYSTEM_PROMPT)

        removed_fields = {
            "summary",
            "exact_ocr_text",
            "ocr_blocks",
            "terminal_or_error_text",
            "code_or_command_text",
            "ui_text",
            "direct_visual_observations",
            "task_relevant_visual_reasoning",
            "candidate_visual_answer",
            "important_details",
        }
        self.assertTrue(removed_fields.isdisjoint(media_item))

    def test_prompt_version_invalidates_previous_analysis_cache(self) -> None:
        self.assertEqual(
            ANALYZER_PROMPT_VERSION,
            "official-transparent-v7-evidence-blocks",
        )

    def test_chat_analyzer_request_contains_exact_request_and_new_contract(self) -> None:
        original = {
            "model": "deepseek-v4-mm-bridge",
            "messages": [
                {
                    "role": "user",
                    "content": "Explain why the Save button is disabled.",
                }
            ],
        }
        image_block = {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,AAAA"},
        }
        item = MediaItem(
            index=1,
            kind="image",
            path="messages[0].content[1]",
            block=image_block,
            approx_bytes=3,
            hash="test-hash",
        )

        body = build_analyzer_body(
            original,
            "/v1/chat/completions",
            [item],
            Settings(),
            "qwen-vision",
        )

        self.assertEqual(body["messages"][0]["content"], ANALYZER_SYSTEM_PROMPT)
        analyzer_content = body["messages"][1]["content"]
        instruction = analyzer_content[0]["text"]
        self.assertIn("Explain why the Save button is disabled.", instruction)
        self.assertIn("text_blocks", instruction)
        self.assertIn("visual_blocks", instruction)
        self.assertIn("audio_blocks", instruction)
        self.assertNotIn("candidate_visual_answer", instruction)
        self.assertEqual(analyzer_content[1], image_block)

    def test_final_context_explains_new_evidence_fields(self) -> None:
        context = build_final_context('{"media": []}')

        self.assertIn("text_blocks[].text", context)
        self.assertIn('basis="observed"', context)
        self.assertIn('basis="inferred"', context)
        self.assertIn("audio_blocks", context)
        self.assertNotIn("exact_ocr_text", context)


if __name__ == "__main__":
    unittest.main()
