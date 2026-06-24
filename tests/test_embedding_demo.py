import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from embedding_demo import (
    build_model_catalog,
    build_query_demo,
    chunk_markdown_text,
    load_extracted_chunks,
    load_markdown_chunks,
    render_markdown_report,
)


REPO_ROOT = Path(__file__).resolve().parent.parent


def create_sample_markdown() -> str:
    return "# 제목\n\n3GPP 임베딩 비교 데모 문서입니다. Qwen과 OpenAI 비교를 포함합니다.\n\nColBERT와 reranker 후보군 확인."


class EmbeddingDemoTests(unittest.TestCase):
    def test_model_catalog_contains_requested_targets(self) -> None:
        """Verify the demo exposes all requested models and the openai_compatible_api/local_cpu modes."""
        models = build_model_catalog()
        model_names = {model.display_name for model in models}
        self.assertEqual(
            model_names,
            {
                "text-small (OpenAI)",
                "Qwen/Qwen3-Embedding-8B",
                "bge-m3",
                "bge-reranker-v2-m3",
                "ColBERT",
                "Nemotron",
            },
        )
        runtime_modes = {model.runtime_mode for model in models}
        self.assertEqual(runtime_modes, {"openai_compatible_api", "local_cpu"})

    def test_markdown_and_extracted_inputs_are_chunked(self) -> None:
        markdown_chunks = chunk_markdown_text(
            create_sample_markdown(),
            str(Path(tempfile.gettempdir()) / "demo.md"),
            max_chars=50,
        )
        self.assertGreaterEqual(len(markdown_chunks), 3)

        extracted_chunks = load_extracted_chunks(inline_items=["중요 문장 1", "중요 문장 2"])
        self.assertEqual([chunk.text for chunk in extracted_chunks], ["중요 문장 1", "중요 문장 2"])

    def test_query_demo_returns_top_k_per_model(self) -> None:
        """Verify query-driven output structure, Top K count per model, and markdown rendering."""
        chunks = load_extracted_chunks(
            inline_items=[
                "3GPP 임베딩 비교를 위한 OpenAI text-small 기준 문장",
                "일반적인 unrelated 문장",
                "Qwen 임베딩과 ColBERT 후보 비교 문장",
            ]
        )
        result = build_query_demo("OpenAI 임베딩 비교", chunks, top_k=2)
        model_count = len(build_model_catalog())

        self.assertEqual(result["query"], "OpenAI 임베딩 비교")
        self.assertEqual(result["top_k"], 2)
        self.assertEqual(len(result["retrieval_results"]), model_count)
        for model_result in result["retrieval_results"]:
            self.assertEqual(len(model_result["top_k"]), 2)

        top_by_model = {item["model_key"]: item["top_k"][0] for item in result["retrieval_results"]}
        self.assertEqual(top_by_model["openai_text_small"]["chunk_id"], "extract-001")

        markdown_report = render_markdown_report(result)
        self.assertIn("Top K Results By Embedding Method", markdown_report)
        self.assertIn("runtime_mode: openai_compatible_api", markdown_report)
        self.assertIn("runtime_mode: local_cpu", markdown_report)

    def test_json_input_and_cli_output_work(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            markdown_path = temp_path / "demo.md"
            markdown_path.write_text(
                "# Demo\n\n3GPP query review target\n\nColBERT reranking candidate paragraph",
                encoding="utf-8",
            )

            json_path = temp_path / "extract.json"
            json_path.write_text(
                json.dumps([{"text": "OpenAI query match 문장", "source": "llm"}], ensure_ascii=False),
                encoding="utf-8",
            )

            chunks = load_markdown_chunks([str(markdown_path)])
            self.assertEqual(len(chunks), 3)

            extracted_chunks = load_extracted_chunks(json_path=str(json_path))
            self.assertEqual(extracted_chunks[0].source, "llm")

            result = subprocess.run(
                [
                    "python",
                    str(REPO_ROOT / "embedding_demo.py"),
                    "--query",
                    "OpenAI query match",
                    "--docs",
                    str(markdown_path),
                    "--json-input",
                    str(json_path),
                    "--top-k",
                    "2",
                    "--output",
                    "json",
                ],
                check=True,
                capture_output=True,
                text=True,
                cwd=REPO_ROOT,
            )

            payload = json.loads(result.stdout)
            self.assertEqual(payload["input_summary"]["total_chunks"], 4)
            self.assertEqual(payload["top_k"], 2)
            self.assertEqual(len(payload["models"]), len(build_model_catalog()))
            self.assertEqual(len(payload["retrieval_results"][0]["top_k"]), 2)


if __name__ == "__main__":
    unittest.main()
