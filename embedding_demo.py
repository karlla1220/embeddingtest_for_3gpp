from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence


SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
TOKEN_RE = re.compile(r"[0-9A-Za-z가-힣]+")


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    family: str
    provider: str
    dimension_notes: str
    purpose: str
    runtime_mode: str
    runtime_summary: str
    prep_steps: tuple[str, ...]
    review_note: str


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
            runtime_mode="openai_compatible_api",
            runtime_summary="OpenAI-compatible embedding endpoint",
            prep_steps=(
                "Set OPENAI_BASE_URL to the embedding API endpoint.",
                "Set OPENAI_API_KEY for the endpoint.",
                "Set OPENAI_TEXT_SMALL_MODEL to the deployed model name.",
            ),
            review_note="사람이 쿼리 기준으로 상위 K개 문서가 적절히 모이는지 가장 먼저 확인할 기본 dense baseline",
        ),
        ModelSpec(
            key="qwen3_embedding_8b",
            display_name="Qwen/Qwen3-Embedding-8B",
            family="single_embedding",
            provider="Qwen",
            dimension_notes="dimension: high (8,000+)",
            purpose="high-dimensional single-vector baseline",
            runtime_mode="openai_compatible_api",
            runtime_summary="OpenAI-compatible embedding endpoint, typically served through a gateway such as vLLM",
            prep_steps=(
                "Serve the model behind an OpenAI-compatible /embeddings API.",
                "Set QWEN3_EMBEDDING_BASE_URL to that endpoint.",
                "Set QWEN3_EMBEDDING_MODEL to the served model name.",
            ),
            review_note="고차원 dense 임베딩이 같은 쿼리에서 더 관련도 높은 문단을 상단에 배치하는지 비교",
        ),
        ModelSpec(
            key="bge_m3",
            display_name="bge-m3",
            family="single_embedding",
            provider="BAAI",
            dimension_notes="dimension: standard dense (1,024)",
            purpose="general single-vector baseline and candidate generation",
            runtime_mode="local_cpu",
            runtime_summary="Local CPU execution for offline dense retrieval checks",
            prep_steps=(
                "Install a local sentence-transformers style runtime on CPU.",
                "Set BGE_M3_MODEL_PATH to the local model directory or cache key.",
                "Run with CPU-only settings for reproducible offline checks.",
            ),
            review_note="로컬 CPU로 쉽게 반복 확인할 dense 기준선이며 멀티벡터 리랭킹 전 후보 생성 기준으로도 확인",
        ),
        ModelSpec(
            key="bge_reranker_v2_m3",
            display_name="bge-reranker-v2-m3",
            family="cross_encoder_reranker",
            provider="BAAI",
            dimension_notes="dimension: not applicable",
            purpose="cross-encoder reranking of retrieved candidates",
            runtime_mode="local_cpu",
            runtime_summary="Local CPU reranker for manual Top K inspection after candidate generation",
            prep_steps=(
                "Install a local cross-encoder runtime on CPU.",
                "Set BGE_RERANKER_MODEL_PATH to the reranker directory or cache key.",
                "Feed it the first-stage Top K candidates for final inspection.",
            ),
            review_note="실서비스에서는 dense 후보군 재정렬용이지만 데모에서는 전 문서 기준 점수도 바로 확인 가능",
        ),
        ModelSpec(
            key="colbert",
            display_name="ColBERT",
            family="late_interaction",
            provider="Stanford",
            dimension_notes="dimension: multi-vector token embeddings",
            purpose="late-interaction reranking baseline compatible with Qdrant flows",
            runtime_mode="local_cpu",
            runtime_summary="Local CPU late-interaction preparation for manual reranking review",
            prep_steps=(
                "Prepare a local ColBERT-style index or token embedding cache.",
                "Set COLBERT_INDEX_PATH to the local index location.",
                "Use CPU-only search for deterministic reviewer checks on a small corpus.",
            ),
            review_note="멀티벡터 late interaction이 정답 문서를 더 위로 끌어올리는지 사람이 직접 보기 좋음",
        ),
        ModelSpec(
            key="nemotron",
            display_name="Nemotron",
            family="late_interaction",
            provider="NVIDIA",
            dimension_notes="dimension: multi-vector token embeddings",
            purpose="late-interaction comparison target",
            runtime_mode="local_cpu",
            runtime_summary="Local CPU late-interaction comparison path for small manual review sets",
            prep_steps=(
                "Prepare a local Nemotron embedding or rerank artifact on CPU.",
                "Set NEMOTRON_MODEL_PATH to the local artifact directory.",
                "Use it on small reviewer-selected candidate sets for qualitative comparison.",
            ),
            review_note="ColBERT와 함께 멀티벡터 계열이 dense baseline보다 상위 K 품질을 얼마나 끌어올리는지 비교",
        ),
    ]


def _normalize_blocks(markdown_text: str) -> list[str]:
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

    sentence_chunks: list[str] = []
    current = ""

    for sentence in sentences:
        candidate = f"{current} {sentence}".strip() if current else sentence
        if current and len(candidate) > max_chars:
            sentence_chunks.append(current)
            current = sentence
            continue
        if len(sentence) > max_chars:
            if current:
                sentence_chunks.append(current)
                current = ""
            sentence_chunks.extend(
                piece.strip()
                for piece in (sentence[i : i + max_chars] for i in range(0, len(sentence), max_chars))
                if piece.strip()
            )
            continue
        current = candidate

    if current:
        sentence_chunks.append(current)

    return sentence_chunks


def chunk_markdown_text(markdown_text: str, source: str, max_chars: int = 400) -> list[Chunk]:
    chunk_texts: list[str] = []
    for block in _normalize_blocks(markdown_text):
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


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def _ordered_bigram_overlap(query_tokens: Sequence[str], chunk_tokens: Sequence[str]) -> float:
    query_bigrams = list(zip(query_tokens, query_tokens[1:]))
    if not query_bigrams:
        return 0.0
    chunk_bigram_set = set(zip(chunk_tokens, chunk_tokens[1:]))
    matches = sum(1 for bigram in query_bigrams if bigram in chunk_bigram_set)
    return matches / len(query_bigrams)


def _phrase_bonus(query: str, chunk_text: str) -> float:
    normalized_query = " ".join(query.lower().split())
    normalized_chunk = " ".join(chunk_text.lower().split())
    if not normalized_query or not normalized_chunk:
        return 0.0
    if normalized_query in normalized_chunk:
        return 1.0
    compact_query = normalized_query.replace(" ", "")
    compact_chunk = normalized_chunk.replace(" ", "")
    return 0.5 if compact_query and compact_query in compact_chunk else 0.0


def score_chunk(query: str, chunk: Chunk, model: ModelSpec) -> tuple[float, list[str]]:
    """Return a deterministic demo score and matched query terms for one chunk/model pair.

    The score is a lightweight lexical surrogate so this repository stays dependency-free:
    query-term coverage, chunk density, ordered bigram overlap, phrase containment, and a
    small heading bonus are combined with model-specific weights. The return value is a
    tuple of `(score, matched_terms)` so the caller can both rank results and explain why
    a chunk was surfaced in the Top K output.
    """
    query_tokens = _tokenize(query)
    chunk_tokens = _tokenize(chunk.text)
    query_terms = set(query_tokens)
    chunk_terms = set(chunk_tokens)
    common_terms = sorted(query_terms & chunk_terms)

    if not query_terms or not chunk_terms:
        return 0.0, common_terms

    coverage = len(common_terms) / len(query_terms)
    density = len(common_terms) / len(chunk_terms)
    ordered_overlap = _ordered_bigram_overlap(query_tokens, chunk_tokens)
    phrase = _phrase_bonus(query, chunk.text)
    heading_bonus = 0.35 if chunk.text.lstrip().startswith("#") else 0.0

    # Tuple order: coverage_weight, ordered_weight, phrase_weight, density_weight, heading_weight.
    # The values intentionally emphasize exact or near-exact matches more strongly for rerankers
    # and late-interaction models so reviewers can inspect different Top K orderings per method.
    score_profiles = {
        "openai_text_small": (3.0, 1.2, 1.5, 0.4, 0.0),
        "qwen3_embedding_8b": (3.2, 1.0, 1.8, 0.3, 0.1),
        "bge_m3": (3.1, 1.3, 1.4, 0.2, 0.0),
        "bge_reranker_v2_m3": (3.5, 2.4, 2.2, 0.4, 0.0),
        "colbert": (3.4, 1.9, 1.8, 0.2, 0.2),
        "nemotron": (3.3, 2.0, 1.9, 0.2, 0.15),
    }
    coverage_weight, ordered_weight, phrase_weight, density_weight, heading_weight = score_profiles[model.key]

    score = (
        (coverage * coverage_weight)
        + (ordered_overlap * ordered_weight)
        + (phrase * phrase_weight)
        + (density * density_weight)
        + (heading_bonus * heading_weight)
    )

    # Non-dense-only models receive a small extra boost for multiple matched terms because the
    # demo is trying to approximate rerank behavior on the same review corpus without real inference.
    if model.family != "single_embedding":
        score += min(len(common_terms), 4) * 0.08

    return round(score, 6), common_terms


def _summarize_text(text: str, max_chars: int = 160) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 1].rstrip() + "…"


def rank_chunks_for_model(query: str, chunks: Sequence[Chunk], model: ModelSpec, top_k: int) -> dict[str, object]:
    ranked = []
    for chunk in chunks:
        score, match_terms = score_chunk(query, chunk, model)
        ranked.append(
            {
                "chunk_id": chunk.chunk_id,
                "source": chunk.source,
                "kind": chunk.kind,
                "score": score,
                "match_terms": match_terms,
                "text": chunk.text,
                "preview": _summarize_text(chunk.text),
            }
        )

    ranked.sort(key=lambda item: (-item["score"], item["chunk_id"]))
    top_results = []
    for index, item in enumerate(ranked[:top_k], start=1):
        top_results.append(
            {
                "rank": index,
                "chunk_id": item["chunk_id"],
                "source": item["source"],
                "kind": item["kind"],
                "score": item["score"],
                "match_terms": item["match_terms"],
                "preview": item["preview"],
            }
        )

    return {
        "model_key": model.key,
        "display_name": model.display_name,
        "family": model.family,
        "runtime_mode": model.runtime_mode,
        "runtime_summary": model.runtime_summary,
        "review_note": model.review_note,
        "top_k": top_results,
    }


def build_query_demo(query: str, chunks: Sequence[Chunk], top_k: int) -> dict[str, object]:
    """Build the query-review payload used by both JSON output and markdown rendering.

    The returned mapping contains top-level query/setup metadata, the serialized model
    catalog with runtime preparation details, the per-model `retrieval_results` Top K list,
    and the raw chunk corpus so a caller can inspect or post-process the same ranking input.
    """
    catalog = build_model_catalog()
    return {
        "query": query,
        "top_k": top_k,
        "review_goal": "사람이 각 임베딩 방식별 Top K를 직접 보고, 관련 문서가 최상단에 잘 오는지 확인",
        "input_summary": {
            "total_chunks": len(chunks),
            "sources": sorted({chunk.source for chunk in chunks}),
            "chunk_kinds": sorted({chunk.kind for chunk in chunks}),
        },
        "models": [asdict(model) for model in catalog],
        "retrieval_results": [rank_chunks_for_model(query, chunks, model, top_k) for model in catalog],
        "chunks": [asdict(chunk) for chunk in chunks],
    }


def render_markdown_report(result: dict[str, object]) -> str:
    input_summary = result.get("input_summary") or {}
    models: list[dict[str, object]] = result.get("models") or []
    retrieval_results: list[dict[str, object]] = result.get("retrieval_results") or []
    sources = input_summary.get("sources") if isinstance(input_summary, dict) else []
    chunk_kinds = input_summary.get("chunk_kinds") if isinstance(input_summary, dict) else []
    total_chunks = input_summary.get("total_chunks") if isinstance(input_summary, dict) else 0

    lines = [
        "# Embedding Comparison Demo",
        "",
        "## Review Goal",
        str(result.get("review_goal", "not provided")),
        "",
        "## Query Setup",
        f"- Query: {result.get('query', '')}",
        f"- Top K: {result.get('top_k', 0)}",
        f"- Total chunks: {total_chunks}",
        f"- Sources: {', '.join(sources) if sources else 'none'}",
        f"- Chunk kinds: {', '.join(chunk_kinds) if chunk_kinds else 'none'}",
        "",
        "## Runtime Preparation",
    ]

    for model in models:
        lines.append(
            f"### {model.get('display_name', 'unknown')}"
        )
        lines.append(f"- family: {model.get('family', 'unknown')}")
        lines.append(f"- runtime_mode: {model.get('runtime_mode', 'unknown')}")
        lines.append(f"- runtime_summary: {model.get('runtime_summary', 'unknown')}")
        lines.append(f"- dimension_notes: {model.get('dimension_notes', 'unknown')}")
        prep_steps = model.get("prep_steps", [])
        for step in prep_steps:
            lines.append(f"  - {step}")
        lines.append("")

    lines.append("## Top K Results By Embedding Method")
    for item in retrieval_results:
        lines.append(f"### {item.get('display_name', 'unknown')}")
        lines.append(f"- family: {item.get('family', 'unknown')}")
        lines.append(f"- runtime_mode: {item.get('runtime_mode', 'unknown')}")
        lines.append(f"- review_note: {item.get('review_note', 'unknown')}")
        for ranked in item.get("top_k", []):
            lines.append(
                f"- #{ranked.get('rank')} | score={ranked.get('score')} | "
                f"{ranked.get('chunk_id')} | matches={', '.join(ranked.get('match_terms', [])) or 'none'}"
            )
            lines.append(f"  - source: {ranked.get('source', 'unknown')}")
            lines.append(f"  - preview: {ranked.get('preview', '')}")
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Show per-query Top K retrieval results across embedding methods.")
    parser.add_argument("--query", required=True, help="Natural language query to inspect.")
    parser.add_argument("--docs", nargs="*", default=[], help="Absolute paths to markdown files or directories.")
    parser.add_argument("--json-input", help="Absolute path to a JSON file of extracted sentences.")
    parser.add_argument(
        "--inline-item",
        action="append",
        default=[],
        help="Repeatable inline extracted sentence input.",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of ranked items to show per model.")
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
    result = build_query_demo(args.query, chunks, args.top_k)

    if args.output == "json":
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(render_markdown_report(result), end="")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
