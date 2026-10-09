"""
Evaluation prompt templates for LLM-based essay grading.
These prompts are designed for Arabic-language essay evaluation
with rubric-aware structured output.
"""

from src.evaluation.rubric import GradingConfig
from src.evaluation.schemas import KnowledgeConfigRequest


# System prompt that frames the evaluation task
EVALUATION_SYSTEM_PROMPT = """أنت مصحح أكاديمي خبير ومحايد. مهمتك تقييم إجابة الطالب بناءً على المعايير (Rubric) المحددة بدقة.

قواعد صارمة:
1. قيّم فقط بناءً على المعايير المقدمة — لا تمنح درجات بناءً على معرفة خارجية أو انطباع عام
2. كل معيار يُقيّم مستقلاً — الدرجة النهائية هي مجموع درجات المعايير فقط
3. الدليل (evidence) يجب أن يكون نصاً مقتبساً حرفياً من إجابة الطالب يثبت تحقيق المعيار
4. إذا لم يحقق الطالب معياراً، أعطِ درجة 0 مع evidence_type = "ABSENT"
5. لا تخترع معلومات — إذا لم يكن في الإجابة ما يدعم المعيار، الدرجة 0
6. كن عادلاً وموضوعياً — نفس المعايير تطبق على جميع الطلاب

صيغة الإخراج: JSON فقط، بالشكل المحدد أدناه. لا تضف أي نص خارج JSON."""


def build_evaluation_prompt(
    question_text: str,
    student_answer: str,
    grading: GradingConfig,
    context: str = "",
) -> str:
    """
    Build the complete evaluation prompt for the LLM.
    
    Args:
        question_text: The question prompt
        student_answer: Student's response text
        grading: GradingConfig with reference answer, max_score, criteria
        context: Retrieved knowledge context (optional)
    
    Returns:
        Complete prompt string for LLM
    """
    # Build criteria JSON for the prompt
    criteria_json = [
        {
            "id": c.id,
            "description": c.description,
            "max_score": c.max_score,
        }
        for c in grading.criteria
    ]

    # Format criteria for readability in prompt
    criteria_lines = []
    for c in grading.criteria:
        criteria_lines.append(
            f'  - {c.id}: {c.description} (max: {c.max_score})'
        )
    criteria_text = "\n".join(criteria_lines)

    prompt = f"""{EVALUATION_SYSTEM_PROMPT}

=== السؤال ===
{question_text}

=== الإجابة النموذجية (Reference Answer) ===
{grading.reference_answer}

=== إجابة الطالب ===
{student_answer}

=== المعايير (Rubric) ===
إجمالي الدرجة: {grading.max_score}
المعايير:
{criteria_text}

=== السياق المرجعي (Retrieved Knowledge) ===
{context if context else "(لا يوجد سياق مرجعي متاح)"}

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

    return prompt


def build_context_from_chunks(chunks: list[dict], max_chars: int = 3000) -> str:
    """
    Build context string from retrieved chunks for the evaluation prompt.
    
    Args:
        chunks: List of retrieved chunk payloads with 'raw_text' and 'chunk_id'
        max_chars: Maximum characters for context
    
    Returns:
        Formatted context string
    """
    if not chunks:
        return ""
    
    context_parts = []
    total_chars = 0
    
    for chunk in chunks:
        text = chunk.get("raw_text", "")
        chunk_id = chunk.get("chunk_id", "unknown")
        part = f"[{chunk_id}] {text}"
        
        if total_chars + len(part) > max_chars:
            break
            
        context_parts.append(part)
        total_chars += len(part)
    
    return "\n\n".join(context_parts)


def parse_llm_evaluation_output(raw_output: str) -> dict:
    """
    Parse LLM output as JSON, with error handling.
    
    Args:
        raw_output: Raw string from LLM
    
    Returns:
        Parsed dictionary
    
    Raises:
        ValueError: If output is not valid JSON or missing required fields
    """
    import json
    
    # Try to extract JSON from output (LLM might add explanation)
    try:
        return json.loads(raw_output)
    except json.JSONDecodeError:
        # Try to find JSON block
        import re
        json_match = re.search(r'\{.*\}', raw_output, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        raise ValueError(f"LLM output is not valid JSON: {raw_output[:200]}")


# Example usage for testing
if __name__ == "__main__":
    # Test prompt building
    grading = GradingConfig(
        reference_answer="ثورة 1919 قامت بسبب الاستعمار البريطاني...",
        max_score=10,
        rubric_version="v1",
        criteria=[
            Criterion(id="c1", description="الأسباب السياسية", max_score=4),
            Criterion(id="c2", description="الأسباب الاقتصادية", max_score=3),
            Criterion(id="c3", description="الأسباب الاجتماعية", max_score=3),
        ]
    )
    
    prompt = build_evaluation_prompt(
        question_text="اشرح أسباب ثورة 1919.",
        student_answer="ثورة 1919 قامت بسبب الاستعمار البريطاني...",
        grading=grading,
        context="[doc1] ثورة 1919 هي ثورة شعبية..."
    )
    print(prompt)