# embeddingtest_for_3gpp

간단한 임베딩 비교 데모를 위해 `embedding_demo.py`를 추가했습니다.

이 데모는 다음을 바로 준비합니다.

- 비교 대상 모델 카탈로그
  - `text-small (OpenAI)`
  - `Qwen/Qwen3-Embedding-8B`
  - `bge-m3`
  - `bge-reranker-v2-m3`
  - `ColBERT`
  - `Nemotron`
- Markdown 문서와 별도 추출 문장(JSON / inline text) 입력
- 문장/문단 단위 청크 생성
- 검증 질문별 실험 계획과 추천 지표 출력

## 사용 예시

```bash
python embedding_demo.py \
  --docs /absolute/path/to/docs \
  --json-input /absolute/path/to/extracted_sentences.json
```

```bash
python embedding_demo.py \
  --docs /absolute/path/to/file.md \
  --inline-item "핵심 문장 1" \
  --inline-item "핵심 문장 2" \
  --output json
```

## JSON 입력 형식

다음 세 형식 중 하나를 지원합니다.

```json
["문장 1", "문장 2"]
```

```json
{
  "items": ["문장 1", "문장 2"]
}
```

```json
[
  {"text": "문장 1", "source": "llm_extract"},
  {"text": "문장 2"}
]
```

## 테스트

```bash
python -m unittest discover -s tests
```