from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence


SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    family: str
    provider: str
    dimension_notes: str
    purpose: str


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    source: str
    kind: str
    text: str


def build_model_catalog() -> list[ModelSpec]:
    return [
        ModelSpec(
            key="openai_text_small",
            display_name="text-small (OpenAI)",
            family="single_embedding",
            provider="OpenAI",
            dimension_notes="dimension: medium-low (~1,500)",
            purpose="single-vector baseline and first-stage retrieval",
        ),
        ModelSpec(
            key="qwen3_embedding_8b",
            display_name="Qwen/Qwen3-Embedding-8B",
            family="single_embedding",
            provider="Qwen",
            dimension_notes="dimension: high (8,000+)",
            purpose="high-dimensional single-vector baseline",
        ),
        ModelSpec(
            key="bge_m3",
            display_name="bge-m3",
            family="single_embedding",
            provider="BAAI",
            dimension_notes="dimension: standard dense (1,024)",
            purpose="general single-vector baseline and candidate generation",
        ),
        ModelSpec(
            key="bge_reranker_v2_m3",
            display_name="bge-reranker-v2-m3",
            family="cross_encoder_reranker",
            provider="BAAI",
            dimension_notes="dimension: not applicable",
            purpose="cross-encoder reranking of retrieved candidates",
        ),
        ModelSpec(
            key="colbert",
            display_name="ColBERT",
            family="late_interaction",
            provider="Stanford",
            dimension_notes="dimension: multi-vector token embeddings",
            purpose="late-interaction reranking baseline compatible with Qdrant flows",
        ),
        ModelSpec(
            key="nemotron",
            display_name="Nemotron",
            family="late_interaction",
            provider="NVIDIA",
            dimension_notes="dimension: multi-vector token embeddings",
            purpose="late-interaction comparison target",
        ),
    ]


def _normalise_blocks(markdown_text: str) -> list[str]:
    lines = [line.rstrip() for line in markdown_text.splitlines()]
    blocks: list[str] = []
    current: list[str] = []

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            if current:
                blocks.append(" ".join(current).strip())
                current = []
            continue
        if line.startswith("#") and current:
            blocks.append(" ".join(current).strip())
            current = [line]
            continue
        current.append(line)

    if current:
        blocks.append(" ".join(current).strip())

    return [block for block in blocks if block]


def _split_oversized_text(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]

    sentences = [sentence.strip() for sentence in SENTENCE_SPLIT_RE.split(text) if sentence.strip()]
    if not sentences:
        words = text.split()
        if words:
            word_chunks: list[str] = []
            current = ""
            for word in words:
                candidate = f"{current} {word}".strip() if current else word
                if current and len(candidate) > max_chars:
                    word_chunks.append(current)
                    current = word
                    continue
                if len(word) > max_chars:
                    if current:
                        word_chunks.append(current)
                        current = ""
                    word_chunks.extend(
                        piece.strip()
                        for piece in (word[i : i + max_chars] for i in range(0, len(word), max_chars))
                        if piece.strip()
                    )
                    continue
                current = candidate
            if current:
                word_chunks.append(current)
            return word_chunks

        return [text[i : i + max_chars].strip() for i in range(0, len(text), max_chars)]

    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        candidate = f"{current} {sentence}".strip() if current else sentence
        if current and len(candidate) > max_chars:
            chunks.append(current)
            current = sentence
            continue
        if len(sentence) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(
                piece.strip()
                for piece in (sentence[i : i + max_chars] for i in range(0, len(sentence), max_chars))
                if piece.strip()
            )
            continue
        current = candidate

    if current:
        chunks.append(current)

    return chunks


def chunk_markdown_text(markdown_text: str, source: str, max_chars: int = 400) -> list[Chunk]:
    chunk_texts: list[str] = []
    for block in _normalise_blocks(markdown_text):
        chunk_texts.extend(_split_oversized_text(block, max_chars=max_chars))

    return [
        Chunk(
            chunk_id=f"{Path(source).stem or 'input'}-{index:03d}",
            source=source,
            kind="markdown",
            text=text,
        )
        for index, text in enumerate(chunk_texts, start=1)
    ]


def discover_markdown_files(paths: Sequence[str]) -> list[Path]:
    discovered: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        if path.is_dir():
            discovered.extend(sorted(candidate for candidate in path.rglob("*.md") if candidate.is_file()))
            continue
        if path.is_file() and path.suffix.lower() == ".md":
            discovered.append(path)
    return discovered


def load_markdown_chunks(paths: Sequence[str], max_chars: int = 400) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in discover_markdown_files(paths):
        chunks.extend(chunk_markdown_text(path.read_text(encoding="utf-8"), str(path), max_chars=max_chars))
    return chunks


def _coerce_extracted_items(payload: object) -> list[dict[str, str]]:
    if isinstance(payload, dict):
        payload = payload.get("items", [])

    if not isinstance(payload, list):
        raise ValueError(
            "Extracted sentence input must be a list or an object with an 'items' list; "
            f"received {type(payload).__name__}."
        )

    items: list[dict[str, str]] = []
    for index, entry in enumerate(payload, start=1):
        if isinstance(entry, str):
            text = entry.strip()
            source = "json_extract"
        elif isinstance(entry, dict):
            text = str(entry.get("text", "")).strip()
            source = str(entry.get("source", "json_extract")).strip() or "json_extract"
        else:
            raise ValueError(f"Unsupported extracted entry at position {index}: {type(entry).__name__}")

        if text:
            items.append({"text": text, "source": source})

    return items


def load_extracted_chunks(*, json_path: str | None = None, inline_items: Sequence[str] = ()) -> list[Chunk]:
    items: list[dict[str, str]] = []

    if json_path:
        payload = json.loads(Path(json_path).expanduser().resolve().read_text(encoding="utf-8"))
        items.extend(_coerce_extracted_items(payload))

    items.extend(
        {"text": text.strip(), "source": "inline_extract"}
        for text in inline_items
        if text and text.strip()
    )

    return [
        Chunk(
            chunk_id=f"extract-{index:03d}",
            source=item["source"],
            kind="extracted",
            text=item["text"],
        )
        for index, item in enumerate(items, start=1)
    ]


def build_demo_plan(chunks: Sequence[Chunk]) -> dict[str, object]:
    catalog = build_model_catalog()
    single_models = [model.display_name for model in catalog if model.family == "single_embedding"]
    rerankers = [model.display_name for model in catalog if model.family != "single_embedding"]

    experiments = [
        {
            "name": "single_embedding_baseline",
            "goal": "단일 임베딩만으로 검색 품질이 충분한지 확인",
            "models": single_models,
            "metrics": ["Recall@5", "Recall@10", "MRR", "nDCG@10"],
            "success_criteria": "가장 성능이 좋은 단일 임베딩 모델이 정답 문단을 안정적으로 상위권에 배치",
        },
        {
            "name": "candidate_generation_for_reranking",
            "goal": "단일 임베딩으로 멀티벡터/리랭커용 후보군을 안정적으로 만들 수 있는지 확인",
            "candidate_models": single_models,
            "rerankers": rerankers,
            "candidate_cutoffs": [10, 20, 50],
            "metrics": ["Candidate Recall@10", "Candidate Recall@20", "MRR after rerank"],
            "success_criteria": "단일 임베딩 후보군 안에 정답이 충분히 포함되어 리랭커 성능이 회복 또는 향상",
        },
        {
            "name": "multi_embedding_comparison",
            "goal": "멀티 임베딩 또는 멀티 스테이지 구성이 단일 임베딩 대비 얼마나 개선되는지 확인",
            "strategies": [
                "single best dense model",
                "dense + cross-encoder rerank",
                "dense + late-interaction rerank",
                "union of multiple dense candidate sets",
            ],
            "metrics": ["Recall@10", "MRR", "nDCG@10", "delta vs best single"],
            "success_criteria": "다단계 또는 멀티 임베딩 조합이 가장 성능이 좋은 단일 임베딩 기준선보다 일관되게 향상",
        },
    ]

    return {
        "input_summary": {
            "total_chunks": len(chunks),
            "sources": sorted({chunk.source for chunk in chunks}),
            "chunk_kinds": sorted({chunk.kind for chunk in chunks}),
        },
        "models": [asdict(model) for model in catalog],
        "validation_questions": [
            "단일 임베딩으로 충분한가?",
            "단일 임베딩으로 멀티벡터 리랭킹용 후보를 선정할 수 있을까?",
            "멀티 임베딩이 얼마나 잘 동작하는가?",
        ],
        "experiments": experiments,
        "chunks": [asdict(chunk) for chunk in chunks],
    }


def render_markdown_report(plan: dict[str, object]) -> str:
    input_summary = plan.get("input_summary") or {}
    models: list[dict[str, object]] = plan.get("models") or []
    experiments: list[dict[str, object]] = plan.get("experiments") or []
    questions = plan.get("validation_questions") or []
    sources = input_summary.get("sources") if isinstance(input_summary, dict) else []
    chunk_kinds = input_summary.get("chunk_kinds") if isinstance(input_summary, dict) else []
    total_chunks = input_summary.get("total_chunks") if isinstance(input_summary, dict) else 0

    lines = [
        "# Embedding Comparison Demo",
        "",
        "## Input Summary",
        f"- Total chunks: {total_chunks}",
        f"- Sources: {', '.join(sources) if sources else 'none'}",
        f"- Chunk kinds: {', '.join(chunk_kinds) if chunk_kinds else 'none'}",
        "",
        "## Model Catalog",
    ]

    for model in models:
        lines.append(
            "- "
            f"**{model.get('display_name', 'unknown')}** · "
            f"{model.get('family', 'unknown')} · "
            f"{model.get('dimension_notes', 'unknown')} · "
            f"{model.get('purpose', 'unknown')}"
        )

    lines.extend(["", "## Validation Questions"])
    lines.extend(f"- {question}" for question in questions)
    lines.extend(["", "## Experiment Plan"])

    for experiment in experiments:
        lines.append(f"### {experiment.get('name', 'unnamed_experiment')}")
        lines.append(f"- Goal: {experiment.get('goal', 'not provided')}")
        for key in ("models", "candidate_models", "rerankers", "candidate_cutoffs", "strategies", "metrics"):
            if key in experiment:
                value = experiment[key]
                if isinstance(value, list):
                    value = ", ".join(str(item) for item in value)
                lines.append(f"- {key}: {value}")
        lines.append(f"- success_criteria: {experiment.get('success_criteria', 'not provided')}")
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare an embedding comparison demo plan.")
    parser.add_argument("--docs", nargs="*", default=[], help="Absolute paths to markdown files or directories.")
    parser.add_argument("--json-input", help="Absolute path to a JSON file of extracted sentences.")
    parser.add_argument(
        "--inline-item",
        action="append",
        default=[],
        help="Repeatable inline extracted sentence input.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=400,
        help="Maximum chunk length when splitting markdown blocks.",
    )
    parser.add_argument(
        "--output",
        choices=("markdown", "json"),
        default="markdown",
        help="Output format.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    chunks = load_markdown_chunks(args.docs, max_chars=args.max_chars)
    chunks.extend(load_extracted_chunks(json_path=args.json_input, inline_items=args.inline_item))
    plan = build_demo_plan(chunks)

    if args.output == "json":
        print(json.dumps(plan, ensure_ascii=False, indent=2))
    else:
        print(render_markdown_report(plan), end="")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
