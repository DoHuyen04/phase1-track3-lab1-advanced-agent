from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
from .mock_runtime import FAILURE_MODE_BY_QID, LAST_USAGE, actor_answer, evaluator, reflector
from .schemas import AttemptTrace, QAExample, ReflectionEntry, RunRecord

@dataclass
class BaseAgent:
    agent_type: Literal["react", "reflexion"]
    max_attempts: int = 1

    def _resolve_cost(self, real_tokens: int, real_latency: int, attempt_id: int) -> tuple[int, int]:
        """Dùng số đo THẬT từ LLM; nếu chạy mock (=0) thì rơi về ước lượng cũ."""
        if real_tokens > 0:
            return real_tokens, real_latency
        token_estimate = 320 + (attempt_id * 65) + (120 if self.agent_type == "reflexion" else 0)
        latency_ms = 160 + (attempt_id * 40) + (90 if self.agent_type == "reflexion" else 0)
        return token_estimate, latency_ms

    def _classify_failure(self, example, final_score, judge, traces):
        """Phân loại failure mode THẬT từ kết quả Evaluator (thay vì tra bảng mock).

        Nhờ đó chạy LLM thật sinh ra đa dạng failure mode tự nhiên cho phần Analysis.
        """
        if final_score == 1:
            return "none"
        # Dataset mock: giữ nhãn được thiết kế sẵn để demo deterministic.
        if example.qid in FAILURE_MODE_BY_QID:
            return FAILURE_MODE_BY_QID[example.qid]
        # Looping: reflexion lặp lại cùng một câu trả lời sai qua nhiều lần thử.
        answers = [t.answer.strip().lower() for t in traces if t.answer]
        if len(answers) >= 2 and len(set(answers)) == 1:
            return "looping"
        reason = (judge.reason or "").lower()
        # incomplete_multi_hop: còn thiếu bằng chứng / dừng giữa chuỗi hop.
        if judge.missing_evidence or any(k in reason for k in ("hop", "incomplete", "missing", "partial", "stopped")):
            return "incomplete_multi_hop"
        # entity_drift: chọn nhầm thực thể (có khẳng định sai/thừa).
        if judge.spurious_claims or any(k in reason for k in ("wrong entity", "entity", "drift", "different")):
            return "entity_drift"
        return "wrong_final_answer"

    def run(self, example: QAExample) -> RunRecord:
        reflection_memory: list[str] = []
        reflections: list[ReflectionEntry] = []
        traces: list[AttemptTrace] = []
        final_answer = ""
        final_score = 0
        for attempt_id in range(1, self.max_attempts + 1):
            # --- Actor: trả lời (cộng dồn token + latency THẬT của lần gọi) ---
            answer = actor_answer(example, attempt_id, self.agent_type, reflection_memory)
            attempt_tokens = LAST_USAGE["total_tokens"]
            attempt_latency = LAST_USAGE["latency_ms"]

            # --- Evaluator: chấm điểm ---
            judge = evaluator(example, answer)
            attempt_tokens += LAST_USAGE["total_tokens"]
            attempt_latency += LAST_USAGE["latency_ms"]

            final_answer = answer
            final_score = judge.score

            reflection: ReflectionEntry | None = None
            # Reflexion: nếu sai, là reflexion agent và còn lượt thử -> sinh reflection.
            if judge.score == 0 and self.agent_type == "reflexion" and attempt_id < self.max_attempts:
                reflection = reflector(example, attempt_id, judge)
                attempt_tokens += LAST_USAGE["total_tokens"]
                attempt_latency += LAST_USAGE["latency_ms"]
                reflections.append(reflection)
                # Lưu CẢ lesson lẫn next_strategy vào memory để Actor dùng cho lần sau.
                reflection_memory.append(
                    f"Bài học (lesson): {reflection.lesson}\n"
                    f"Chiến thuật (next_strategy): {reflection.next_strategy}"
                )

            token_estimate, latency_ms = self._resolve_cost(attempt_tokens, attempt_latency, attempt_id)
            trace = AttemptTrace(attempt_id=attempt_id, answer=answer, score=judge.score, reason=judge.reason, reflection=reflection, token_estimate=token_estimate, latency_ms=latency_ms)
            traces.append(trace)

            if judge.score == 1:
                break
        total_tokens = sum(t.token_estimate for t in traces)
        total_latency = sum(t.latency_ms for t in traces)
        failure_mode = self._classify_failure(example, final_score, judge, traces)
        return RunRecord(qid=example.qid, question=example.question, gold_answer=example.gold_answer, agent_type=self.agent_type, predicted_answer=final_answer, is_correct=bool(final_score), attempts=len(traces), token_estimate=total_tokens, latency_ms=total_latency, failure_mode=failure_mode, reflections=reflections, traces=traces)

class ReActAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(agent_type="react", max_attempts=1)

class ReflexionAgent(BaseAgent):
    def __init__(self, max_attempts: int = 3) -> None:
        super().__init__(agent_type="reflexion", max_attempts=max_attempts)
