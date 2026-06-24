import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from embedding_demo import (
    build_demo_plan,
    build_model_catalog,
    chunk_markdown_text,
    load_extracted_chunks,
    load_markdown_chunks,
    render_markdown_report,
)


REPO_ROOT = Path("/home/runner/work/embeddingtest_for_3gpp/embeddingtest_for_3gpp")


class EmbeddingDemoTests(unittest.TestCase):
    def test_model_catalog_contains_requested_targets(self) -> None:
        model_names = {model.display_name for model in build_model_catalog()}
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

    def test_markdown_and_extracted_inputs_are_chunked(self) -> None:
        markdown = "# 제목\n\n첫 문단입니다. 두 번째 문장도 있습니다.\n\n둘째 문단입니다."
        markdown_chunks = chunk_markdown_text(markdown, "/tmp/demo.md", max_chars=30)
        self.assertGreaterEqual(len(markdown_chunks), 3)

        extracted_chunks = load_extracted_chunks(inline_items=["중요 문장 1", "중요 문장 2"])
        self.assertEqual([chunk.text for chunk in extracted_chunks], ["중요 문장 1", "중요 문장 2"])

    def test_demo_plan_includes_required_questions_and_rerank_flow(self) -> None:
        chunks = load_extracted_chunks(inline_items=["핵심 질의 후보"])
        plan = build_demo_plan(chunks)

        self.assertIn("단일 임베딩으로 충분한가?", plan["validation_questions"])
        self.assertIn("단일 임베딩으로 멀티벡터 리랭킹용 후보를 선정할 수 있을까?", plan["validation_questions"])
        self.assertIn("멀티 임베딩이 얼마나 잘 동작하는가?", plan["validation_questions"])

        experiment_names = {experiment["name"] for experiment in plan["experiments"]}
        self.assertIn("candidate_generation_for_reranking", experiment_names)

        markdown_report = render_markdown_report(plan)
        self.assertIn("ColBERT", markdown_report)
        self.assertIn("Nemotron", markdown_report)

    def test_json_input_and_cli_output_work(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            markdown_path = temp_path / "demo.md"
            markdown_path.write_text("# Demo\n\n문단 1\n\n문단 2", encoding="utf-8")

            json_path = temp_path / "extract.json"
            json_path.write_text(
                json.dumps([{"text": "LLM 추출 문장", "source": "llm"}], ensure_ascii=False),
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
                    "--docs",
                    str(markdown_path),
                    "--json-input",
                    str(json_path),
                    "--output",
                    "json",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            payload = json.loads(result.stdout)
            self.assertEqual(payload["input_summary"]["total_chunks"], 4)
            self.assertEqual(len(payload["models"]), 6)


if __name__ == "__main__":
    unittest.main()
