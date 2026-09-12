"""
Generation layer — pluggable, same philosophy as the embedding interface:
tests must be able to exercise the full pipeline without network access,
and swapping the real model in production must be a one-line change, not
a rewrite.
"""

import json
from abc import ABC, abstractmethod


class GenerationResult:
    def __init__(self, answer: str, cited_chunk_ids: list[str]):
        self.answer = answer
        self.cited_chunk_ids = cited_chunk_ids


class LLMClient(ABC):
    name: str

    @abstractmethod
    def generate(self, question: str, context_chunks: list[dict]) -> GenerationResult:
        raise NotImplementedError


GROUNDING_PROMPT_TEMPLATE = """أجب على السؤال بالاعتماد فقط على المقاطع المرجعية أدناه. \
إذا لم تحتوِ المقاطع على إجابة، قل ذلك صراحة ولا تختلق معلومات.

السؤال: {question}

المقاطع المرجعية:
{context}

أعد الإجابة بصيغة JSON فقط بالشكل التالي، من غير أي نص إضافي:
{{"answer": "...", "cited_chunk_ids": ["...", "..."]}}
"""


class AnthropicLLM(LLMClient):
    """Real generation backend. Requires ANTHROPIC_API_KEY and network access."""

    def __init__(self, model: str = "claude-sonnet-4-6"):
        self.name = model

    def generate(self, question: str, context_chunks: list[dict]) -> GenerationResult:
        import os
        import anthropic

        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        context = "\n\n".join(
            f"[{c['chunk_id']}] {c.get('raw_text', '')}" for c in context_chunks
        )
        prompt = GROUNDING_PROMPT_TEMPLATE.format(question=question, context=context)

        response = client.messages.create(
            model=self.name,
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text
        parsed = json.loads(raw)
        return GenerationResult(parsed["answer"], parsed.get("cited_chunk_ids", []))


class FakeLLM(LLMClient):
    """
    Dev-only generation backend — used when no network/API key is
    available (this sandboxed environment). It does NOT generate real
    language — it extracts the top retrieved chunk as the "answer" so the
    query pipeline (retrieval -> context -> generation -> response) can be
    tested end-to-end. Never use this to claim a groundedness/hallucination
    score — those metrics are meaningless without a real LLM call.
    """

    name = "fake-llm-DEV-ONLY"

    def generate(self, question: str, context_chunks: list[dict]) -> GenerationResult:
        if not context_chunks:
            return GenerationResult(
                answer="لا توجد معلومات كافية للإجابة على هذا السؤال.",
                cited_chunk_ids=[],
            )
        top_chunk = context_chunks[0]
        return GenerationResult(
            answer=top_chunk.get("raw_text", ""),
            cited_chunk_ids=[top_chunk["chunk_id"]],
        )
