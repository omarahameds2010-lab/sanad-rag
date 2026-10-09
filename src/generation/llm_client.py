"""
Generation layer — pluggable, same philosophy as the embedding interface:
tests must be able to exercise the full pipeline without network access,
and swapping the real model in production must be a one-line change, not
a rewrite.
"""

import json
import os
from abc import ABC, abstractmethod
from typing import Optional
from pydantic import BaseModel


try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    genai = None
    types = None


class GenerationResult:
    def __init__(self, answer: str, cited_chunk_ids: list[str]):
        self.answer = answer
        self.cited_chunk_ids = cited_chunk_ids


class EvaluationResult(BaseModel):
    """Structured output for essay evaluation."""
    criteria: list[dict]
    overall_feedback_ar: str
    review_signal: float
    knowledge_gaps: list[str] = []


class LLMClient(ABC):
    name: str

    @abstractmethod
    def generate(
        self, 
        question: str, 
        context_chunks: list[dict]
    ) -> GenerationResult:
        """RAG Q&A generation — existing contract."""
        raise NotImplementedError

    @abstractmethod
    def generate_evaluation(
        self,
        prompt: str,
        response_model: Optional[type] = None,
    ) -> str:
        """Structured essay evaluation generation — NEW contract."""
        raise NotImplementedError


GROUNDING_PROMPT_TEMPLATE = """أجب على السؤال بالاعتماد فقط على المقاطع المرجعية أدناه. \
إذا لم تحتوِ المقاطع على إجابة، قل ذلك صراحة ولا تختلق معلومات.

السؤال: {question}

المقاطع المرجعية:
{context}

أعد الإجابة بصيغة JSON فقط بالشكل التالي، من غير أي نص إضافي:
{{"answer": "...", "cited_chunk_ids": ["...", "..."]}}
"""


EVALUATION_PROMPT_TEMPLATE = """أنت مصحح أكاديمي خبير ومحايد. مهمتك تقييم إجابة الطالب بناءً على المعايير (Rubric) المحددة بدقة.

قواعد صارمة:
1. قيّم فقط بناءً على المعايير المقدمة — لا تمنح درجات بناءً على معرفة خارجية أو انطباع عام
2. كل معيار يُقيّم مستقلاً — الدرجة النهائية هي مجموع درجات المعايير فقط
3. الدليل (evidence) يجب أن يكون نصاً مقتبساً حرفياً من إجابة الطالب يثبت تحقيق المعيار
4. إذا لم يحقق الطالب معياراً، أعطِ درجة 0 مع evidence_type = "ABSENT"
5. لا تختلق معلومات — إذا لم يكن في الإجابة ما يدعم المعيار، الدرجة 0
6. كن عادلاً وموضوعياً — نفس المعايير تطبق على جميع الطلاب

صيغة الإخراج: JSON فقط، بالشكل المحدد أدناه. لا تضف أي نص خارج JSON.

=== السؤال ===
{question}

=== الإجابة النموذجية (Reference Answer) ===
{reference_answer}

=== إجابة الطالب ===
{student_answer}

=== المعايير (Rubric) ===
إجمالي الدرجة: {max_score}
المعايير:
{criteria_text}

=== السياق المرجعي (Retrieved Knowledge) ===
{context}

=== تنسيق الإخراج المطلوب ===
أعد التقييم بصيغة JSON فقط بالشكل التالي:
{{
  "criteria": [
    {{
      "criterion_id": "c1",
      "earned": 4,
      "evidence": "النص المقتبس حرفياً من إجابة الطالب",
      "evidence_type": "SUPPORTED",
      "feedback_ar": "تعليق عربي مختصر"
    }}
  ],
  "overall_feedback_ar": "ملخص عام للإجابة",
  "review_signal": 0.85,
  "knowledge_gaps": ["ما ينقص في المراجع إن وجد"]
}}

ملاحظات هامة:
- criterion_id يجب أن يطابق تماماً المعرفات في المعايير أعلاه
- earned يجب أن يكون عدداً صحيحاً بين 0 و max_score للمعيار
- evidence يجب أن يكون نصاً مقتبساً حرفياً من إجابة الطالب، أو null إذا ABSENT
- evidence_type: "SUPPORTED" (دليل واضح)، "PARTIAL" (دليل جزئي)، "ABSENT" (لا دليل)
- review_signal: رقم بين 0.0 و 1.0 يعكس ثقتك بالتقييم
- knowledge_gaps: مصفوفة نصية لأي فجوات معرفية لاحظتها
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

    def generate_evaluation(
        self,
        prompt: str,
        response_model: Optional[type] = None,
    ) -> str:
        """Generate evaluation and return raw string for parsing."""
        import os
        import anthropic

        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        
        # Use a simpler prompt format for evaluation (the prompt already contains full instructions)
        response = client.messages.create(
            model=self.name,
            max_tokens=2000,
            temperature=0.1,  # Low temperature for consistency
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text


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

    def generate_evaluation(
        self,
        prompt: str,
        response_model: Optional[type] = None,
    ) -> str:
        # Dev fallback: return a mock evaluation JSON string for testing
        import json
        return json.dumps({
            "criteria": [
                {
                    "criterion_id": "c1",
                    "earned": 3,
                    "evidence": "نص تجريبي من إجابة الطالب",
                    "evidence_type": "SUPPORTED",
                    "feedback_ar": "تقييم تجريبي",
                }
            ],
            "overall_feedback_ar": "تقييم تجريبي - غير موثوق",
            "review_signal": 0.5,
            "knowledge_gaps": ["هذا تقييم تجريبي فقط"],
        }, ensure_ascii=False)


class GeminiLLM(LLMClient):
    """Google Gemini - Free tier, excellent Arabic, native JSON schema.
    
    Uses the new google-genai SDK (not deprecated google-generativeai).
    Supports both old AIzaSy... and new AQ... API key formats.
    """

    def __init__(self, model: str = "gemini-3.8-flash"):
        try:
            from google import genai
            from google.genai import types
            self._genai = genai
            self._types = types
        except ImportError:
            raise ImportError("google-genai not installed. Run: pip install google-genai")
        
        self.name = model
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set in environment")
        self.client = genai.Client(api_key=api_key)
        self._model = model
    
    def generate(self, question: str, context_chunks: list[dict]) -> GenerationResult:
        from src.generation.llm_client import GenerationResult
        context = "\n\n".join(
            f"[{c['chunk_id']}] {c.get('raw_text', '')}" for c in context_chunks
        )
        prompt = GROUNDING_PROMPT_TEMPLATE.format(question=question, context=context)

        # Bounded call: a hanging provider must not hang the API forever —
        # the query pipeline falls back to FakeLLM on any exception.
        response = _call_with_timeout(
            lambda: self.client.models.generate_content(
                model=self._model,
                contents=prompt,
            ),
            timeout_s=_gemini_timeout(),
        )
        try:
            parsed = json.loads(response.text)
            return GenerationResult(parsed["answer"], parsed.get("cited_chunk_ids", []))
        except (json.JSONDecodeError, KeyError, TypeError):
            # Model returned free text instead of the requested JSON —
            # keep the real answer, just without structured citations.
            return GenerationResult(response.text.strip(), [])
    
    def generate_evaluation(
        self,
        prompt: str,
        response_model: Optional[type] = None,
    ) -> str:
        from google.genai import types

        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.1,
        )
        response = _call_with_timeout(
            lambda: self.client.models.generate_content(
                model=self._model,
                contents=prompt,
                config=config,
            ),
            timeout_s=_gemini_timeout(),
        )
        return response.text


def _gemini_timeout() -> float:
    """Per-call timeout for Gemini (seconds). Configurable via GEMINI_TIMEOUT_S."""
    try:
        return float(os.environ.get("GEMINI_TIMEOUT_S", "45"))
    except ValueError:
        return 45.0


def _call_with_timeout(fn, timeout_s: float):
    """Run a blocking provider call with a hard timeout.

    Raises TimeoutError on expiry so the query pipeline's existing
    primary->fallback logic kicks in instead of hanging the request.
    """
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(fn)
        try:
            return fut.result(timeout=timeout_s)
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"Gemini call exceeded {timeout_s}s")