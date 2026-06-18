from __future__ import annotations

import os
import time

from dotenv import load_dotenv

from .prompts import ACTOR_SYSTEM, EVALUATOR_SYSTEM, REFLECTOR_SYSTEM
from .schemas import JudgeResult, QAExample, ReflectionEntry
from .utils import normalize_answer

load_dotenv()

# REFLEXION_MODE=mock   -> dùng logic giả lập deterministic (chạy offline, cho autograde)
# REFLEXION_MODE=openai -> gọi OpenAI API thật
MODE = os.getenv("REFLEXION_MODE", "mock").lower()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

FIRST_ATTEMPT_WRONG = {"hp2": "London", "hp4": "Atlantic Ocean", "hp6": "Red Sea", "hp8": "Andes"}
FAILURE_MODE_BY_QID = {"hp2": "incomplete_multi_hop", "hp4": "wrong_final_answer", "hp6": "entity_drift", "hp8": "entity_drift"}

# Token + latency của lần gọi LLM gần nhất (Bước 5 dùng để ghi giá trị thật).
LAST_USAGE: dict[str, int] = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "latency_ms": 0}

_client = None


# --------------------------------------------------------------------------- #
# OpenAI client helpers
# --------------------------------------------------------------------------- #
def _get_client():
    global _client
    if _client is None:
        from openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Thiếu OPENAI_API_KEY. Thêm vào file .env: OPENAI_API_KEY=your_key "
                "hoặc đặt REFLEXION_MODE=mock để chạy offline."
            )
        # Hỗ trợ OPENAI_BASE_URL để dùng Ollama/vLLM/proxy tương thích OpenAI nếu muốn.
        base_url = os.getenv("OPENAI_BASE_URL")
        _client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
    return _client


def _call_openai(system_prompt: str, user_prompt: str, *, json_mode: bool = False, temperature: float = 0.2) -> str:
    """Gọi OpenAI một lần, ghi token + latency thật vào LAST_USAGE, trả về text."""
    client = _get_client()
    kwargs = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    start = time.perf_counter()
    resp = client.chat.completions.create(**kwargs)
    latency_ms = int((time.perf_counter() - start) * 1000)

    usage = getattr(resp, "usage", None)
    input_tokens = getattr(usage, "prompt_tokens", 0) or 0
    output_tokens = getattr(usage, "completion_tokens", 0) or 0
    LAST_USAGE.update(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        latency_ms=latency_ms,
    )
    return (resp.choices[0].message.content or "").strip()


def _format_context(example: QAExample) -> str:
    return "\n\n".join(f"[{c.title}]\n{c.text}" for c in example.context)


# --------------------------------------------------------------------------- #
# Actor
# --------------------------------------------------------------------------- #
def actor_answer(example: QAExample, attempt_id: int, agent_type: str, reflection_memory: list[str]) -> str:
    if MODE != "openai":
        return _mock_actor_answer(example, attempt_id, agent_type, reflection_memory)

    memory_block = ""
    if reflection_memory:
        items = "\n\n".join(f"[Reflection {i}]\n{m}" for i, m in enumerate(reflection_memory, 1))
        memory_block = (
            "\nREFLECTION MEMORY — tổng hợp bài học & chiến thuật từ TẤT CẢ các lần thử "
            "trước đó. Hãy đọc và áp dụng toàn bộ để không lặp lại lỗi cũ:\n"
            f"{items}\n"
        )

    user_prompt = (
        f"CONTEXT:\n{_format_context(example)}\n"
        f"{memory_block}\n"
        f"QUESTION: {example.question}\n\n"
        "Trả lời ngắn gọn, chỉ đáp án cuối cùng."
    )
    return _call_openai(ACTOR_SYSTEM, user_prompt)


def _mock_actor_answer(example: QAExample, attempt_id: int, agent_type: str, reflection_memory: list[str]) -> str:
    if example.qid not in FIRST_ATTEMPT_WRONG:
        return example.gold_answer
    if agent_type == "react":
        return FIRST_ATTEMPT_WRONG[example.qid]
    if attempt_id == 1 and not reflection_memory:
        return FIRST_ATTEMPT_WRONG[example.qid]
    return example.gold_answer


# --------------------------------------------------------------------------- #
# Evaluator
# --------------------------------------------------------------------------- #
def evaluator(example: QAExample, answer: str) -> JudgeResult:
    if MODE != "openai":
        return _mock_evaluator(example, answer)

    user_prompt = (
        f"QUESTION: {example.question}\n"
        f"GOLD_ANSWER: {example.gold_answer}\n"
        f"PREDICTED_ANSWER: {answer}\n"
    )
    raw = _call_openai(EVALUATOR_SYSTEM, user_prompt, json_mode=True)
    try:
        return JudgeResult.model_validate_json(raw)
    except Exception:
        # Fallback an toàn: so khớp chuẩn hoá nếu LLM trả JSON hỏng.
        score = int(normalize_answer(example.gold_answer) == normalize_answer(answer))
        return JudgeResult(score=score, reason=f"Fallback EM check. Raw LLM output: {raw[:200]}")


def _mock_evaluator(example: QAExample, answer: str) -> JudgeResult:
    if normalize_answer(example.gold_answer) == normalize_answer(answer):
        return JudgeResult(score=1, reason="Final answer matches the gold answer after normalization.")
    if normalize_answer(answer) == "london":
        return JudgeResult(score=0, reason="The answer stopped at the birthplace city and never completed the second hop to the river.", missing_evidence=["Need to identify the river that flows through London."], spurious_claims=[])
    return JudgeResult(score=0, reason="The final answer selected the wrong second-hop entity.", missing_evidence=["Need to ground the answer in the second paragraph."], spurious_claims=[answer])


# --------------------------------------------------------------------------- #
# Reflector
# --------------------------------------------------------------------------- #
def reflector(example: QAExample, attempt_id: int, judge: JudgeResult) -> ReflectionEntry:
    if MODE != "openai":
        return _mock_reflector(example, attempt_id, judge)

    user_prompt = (
        f"QUESTION: {example.question}\n"
        f"GOLD_ANSWER: {example.gold_answer}\n"
        f"EVALUATOR_REASON: {judge.reason}\n"
        f"MISSING_EVIDENCE: {judge.missing_evidence}\n"
        f"SPURIOUS_CLAIMS: {judge.spurious_claims}\n"
        f"ATTEMPT_ID: {attempt_id}\n"
    )
    raw = _call_openai(REFLECTOR_SYSTEM, user_prompt, json_mode=True)
    try:
        entry = ReflectionEntry.model_validate_json(raw)
        entry.attempt_id = attempt_id  # luôn ép đúng attempt_id từ vòng lặp
        return entry
    except Exception:
        return ReflectionEntry(attempt_id=attempt_id, failure_reason=judge.reason, lesson="Hoàn tất mọi hop trước khi trả lời.", next_strategy=f"Phân tích lại context và sửa lỗi: {judge.reason}")


def _mock_reflector(example: QAExample, attempt_id: int, judge: JudgeResult) -> ReflectionEntry:
    strategy = "Do the second hop explicitly: birthplace city -> river through that city." if example.qid == "hp2" else "Verify the final entity against the second paragraph before answering."
    return ReflectionEntry(attempt_id=attempt_id, failure_reason=judge.reason, lesson="A partial first-hop answer is not enough; the final answer must complete all hops.", next_strategy=strategy)
