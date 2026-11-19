"""Regression coverage for grounded multimodal retrieval and answer shaping."""

from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import Mock, patch

from src.agents.rag_agent import SYSTEM_PROMPT, _merge_evidence, answer_question, has_supported_citation, strip_unsupported_citations
from src.rag.generator import normalize_model_text
from src.rag.retriever import CAPTION_SCHEMA_VERSION, _load_cached_captions, reciprocal_rank_fusion, search_by_image


class CaptionQualityTests(unittest.TestCase):
    def test_repeated_caption_lines_are_removed_and_output_is_bounded(self) -> None:
        caption = """### Éléments visibles
- MatMul
- Scale
- SoftMax
- MatMul
- Linear\n""" + "\n".join(f"Étape {n}" for n in range(20))

        normalized = normalize_model_text(caption, max_chars=300, max_lines=6)

        self.assertEqual(normalized.count("MatMul"), 1)
        self.assertLessEqual(len(normalized.splitlines()), 6)
        self.assertLessEqual(len(normalized), 300)

    def test_repeated_markdown_list_items_are_removed_after_numbering(self) -> None:
        caption = "1. **Linear** projette les vecteurs d'entrée.\n2. **Linear** projette les vecteurs d'entrée."
        self.assertEqual(normalize_model_text(caption), "Linear projette les vecteurs d'entrée.")

    def test_caption_cache_rejects_old_descriptions_for_regeneration(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "captions.json"
            path.write_text(json.dumps([{"filename": "figure.png", "caption": "old caption"}]), encoding="utf-8")
            self.assertEqual(_load_cached_captions(path), {})
            path.write_text(
                json.dumps([{"schema_version": CAPTION_SCHEMA_VERSION, "filename": "figure.png", "caption": "new caption"}]),
                encoding="utf-8",
            )
            self.assertEqual(_load_cached_captions(path), {"figure.png": "new caption"})


class RetrievalQualityTests(unittest.TestCase):
    def test_rank_fusion_combines_order_without_comparing_raw_scores(self) -> None:
        text_hit = {"source_file": "manual.pdf", "page_number": 4, "content_type": "text", "text": "scale softmax", "score": 0.99}
        shared_hit_text = {"source_file": "manual.pdf", "page_number": 4, "content_type": "image", "text": "attention diagram", "score": 0.22}
        shared_hit_image = {**shared_hit_text, "score": 0.91}

        fused = reciprocal_rank_fusion([text_hit, shared_hit_text], [shared_hit_image], limit=5)

        self.assertEqual(fused[0]["content_type"], "image")
        self.assertIsNone(fused[0]["score"])
        self.assertEqual(fused[0]["retrieval_score"], 0.22)
        self.assertGreater(fused[0]["rank_score"], fused[1]["rank_score"])

    def test_photo_and_text_evidence_are_fused_by_rank(self) -> None:
        text_hit = {"source_file": "manual.pdf", "page_number": 2, "content_type": "text", "text": "procedure", "score": 0.4}
        photo_hit = {"source_file": "manual.pdf", "page_number": 4, "content_type": "image", "text": "matching figure", "score": 0.8}
        another_visual_hit = {"source_file": "manual.pdf", "page_number": 2, "content_type": "image", "text": "related figure", "score": 0.2}

        merged = _merge_evidence([text_hit, another_visual_hit], [photo_hit, another_visual_hit])

        self.assertEqual(merged[0]["content_type"], "image")
        self.assertEqual(merged[0]["page_number"], 2)
        self.assertEqual(len(merged), 3)

    def test_photo_search_returns_indexed_page_metadata(self) -> None:
        node = SimpleNamespace(
            metadata={"source_file": "manual.pdf", "page_number": 4, "content_type": "image", "image_path": "figures/page4.png"},
            get_content=lambda metadata_mode="none": "Scaled Dot-Product Attention figure",
        )
        vector_store = Mock()
        vector_store.query.return_value = SimpleNamespace(nodes=[node], similarities=[0.8])
        with patch("src.rag.retriever.get_settings", return_value=SimpleNamespace(clip_model="clip", clip_device="cpu", retrieval_top_k=6)), \
             patch("src.rag.retriever.embed_image", return_value=[0.1, 0.2]) as embed, \
             patch("src.rag.retriever.get_image_vector_store", return_value=vector_store):
            result = search_by_image(b"photo")

        embed.assert_called_once()
        self.assertEqual(result[0]["source_file"], "manual.pdf")
        self.assertEqual(result[0]["page_number"], 4)
        self.assertEqual(result[0]["content_type"], "image")
        self.assertEqual(result[0]["image_path"], "figures/page4.png")


class GroundingTests(unittest.TestCase):
    def test_only_indexed_document_page_citations_survive(self) -> None:
        answer = (
            "La formule applique SoftMax après la mise à l'échelle [manual.pdf, page 4]. "
            "Cette méthode est décrite par Goodfellow (2016). "
            "Voir aussi [other.pdf, page 9]. Les tenseurs sont notés [Q, K, V]."
        )
        sources = [{"source_file": "manual.pdf", "page_number": 4}]

        cleaned = strip_unsupported_citations(answer, sources)

        self.assertIn("[manual.pdf, page 4]", cleaned)
        self.assertIn("[Q, K, V]", cleaned)
        self.assertNotIn("Goodfellow", cleaned)
        self.assertNotIn("other.pdf", cleaned)
        self.assertTrue(has_supported_citation(cleaned, sources))
        self.assertFalse(has_supported_citation("La formule applique SoftMax.", sources))

    def test_no_evidence_abstains_without_calling_generation(self) -> None:
        with patch("src.agents.rag_agent.search_documents", return_value=[]), \
             patch("src.agents.rag_agent.search_by_image") as image_search, \
             patch("src.agents.rag_agent.generate_answer") as generate:
            response = answer_question("Explique cette procédure")

        image_search.assert_not_called()
        generate.assert_not_called()
        self.assertEqual(response["sources"], [])
        self.assertIn("pas trouvé de preuve suffisante", response["answer"])

    def test_prompt_enforces_local_sources_and_attention_distinction(self) -> None:
        self.assertIn("aucun livre, site web, auteur ou référence externe", SYSTEM_PROMPT)
        self.assertIn("Scaled Dot-Product Attention", SYSTEM_PROMPT)
        self.assertIn("Multi-Head Attention", SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
