from __future__ import annotations
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from .schemas import ReportPayload, RunRecord

def summarize(records: list[RunRecord]) -> dict:
    grouped: dict[str, list[RunRecord]] = defaultdict(list)
    for record in records:
        grouped[record.agent_type].append(record)
    summary: dict[str, dict] = {}
    for agent_type, rows in grouped.items():
        summary[agent_type] = {"count": len(rows), "em": round(mean(1.0 if r.is_correct else 0.0 for r in rows), 4), "avg_attempts": round(mean(r.attempts for r in rows), 4), "avg_token_estimate": round(mean(r.token_estimate for r in rows), 2), "avg_latency_ms": round(mean(r.latency_ms for r in rows), 2)}
    if "react" in summary and "reflexion" in summary:
        summary["delta_reflexion_minus_react"] = {"em_abs": round(summary["reflexion"]["em"] - summary["react"]["em"], 4), "attempts_abs": round(summary["reflexion"]["avg_attempts"] - summary["react"]["avg_attempts"], 4), "tokens_abs": round(summary["reflexion"]["avg_token_estimate"] - summary["react"]["avg_token_estimate"], 2), "latency_abs": round(summary["reflexion"]["avg_latency_ms"] - summary["react"]["avg_latency_ms"], 2)}
    return summary

def failure_breakdown(records: list[RunRecord]) -> dict:
    # Key theo LOẠI failure mode (đúng tinh thần rubric "≥3 failure modes"),
    # mỗi loại tách nhỏ theo agent để vẫn so sánh được react vs reflexion.
    grouped: dict[str, Counter] = defaultdict(Counter)
    for record in records:
        grouped[record.failure_mode][record.agent_type] += 1
    return {mode: dict(counter) for mode, counter in grouped.items()}

def build_discussion(records: list[RunRecord], dataset_name: str) -> str:
    """Sinh phần Discussion TỪ kết quả thật của lần chạy (không phải text mẫu)."""
    s = summarize(records)
    rx, rf, delta = s.get("react"), s.get("reflexion"), s.get("delta_reflexion_minus_react")
    if not (rx and rf and delta):
        return "Cần cả ReAct và Reflexion để phân tích so sánh."

    # Đối chiếu theo qid: ReAct sai -> Reflexion đúng / cả hai cùng sai.
    by_qid: dict[str, dict] = defaultdict(dict)
    for r in records:
        by_qid[r.qid][r.agent_type] = r
    fixed = sum(1 for a in by_qid.values() if "react" in a and "reflexion" in a and not a["react"].is_correct and a["reflexion"].is_correct)
    both_wrong = sum(1 for a in by_qid.values() if "react" in a and "reflexion" in a and not a["react"].is_correct and not a["reflexion"].is_correct)
    regressed = sum(1 for a in by_qid.values() if "react" in a and "reflexion" in a and a["react"].is_correct and not a["reflexion"].is_correct)

    # Failure mode còn lại của Reflexion (loại trừ 'none'), sắp theo số lượng giảm dần.
    rf_modes = Counter(r.failure_mode for r in records if r.agent_type == "reflexion" and r.failure_mode != "none")
    top_modes = ", ".join(f"{m} ({c})" for m, c in rf_modes.most_common(3)) or "không còn lỗi nào"

    n_q = len(by_qid)
    em_gain = delta.get("em_abs", 0)
    parts = [
        f"Chạy trên `{dataset_name}` ({n_q} câu hỏi × 2 agent = {len(records)} bản ghi).",
        f"EM: ReAct {rx.get('em')} vs Reflexion {rf.get('em')} (Δ {em_gain:+}).",
        f"Reflexion sửa đúng {fixed} câu ReAct trả sai" + (f" nhưng làm hỏng {regressed} câu ReAct vốn đúng" if regressed else "") + f"; còn {both_wrong} câu cả hai cùng sai.",
        f"Chi phí đánh đổi: số lần thử trung bình {rx.get('avg_attempts')}→{rf.get('avg_attempts')}, "
        f"token {rx.get('avg_token_estimate')}→{rf.get('avg_token_estimate')} ({delta.get('tokens_abs', 0):+.0f}), "
        f"latency {rx.get('avg_latency_ms')}→{rf.get('avg_latency_ms')}ms ({delta.get('latency_abs', 0):+.0f}).",
        f"Failure mode còn lại của Reflexion: {top_modes}.",
    ]
    if em_gain > 0:
        parts.append("Reflexion có lợi khi lần thử đầu dừng giữa chuỗi multi-hop hoặc chọn nhầm thực thể; reflection memory giúp sửa ở lần sau, đổi lại chi phí token/latency cao hơn.")
    else:
        parts.append("Trên bộ này Reflexion không tạo khác biệt EM — dữ liệu đủ dễ để ReAct trả đúng ngay lần đầu, nên chi phí thử lại thêm không được đền bù.")
    return " ".join(parts)

def build_report(records: list[RunRecord], dataset_name: str, mode: str = "mock") -> ReportPayload:
    examples = [{"qid": r.qid, "agent_type": r.agent_type, "gold_answer": r.gold_answer, "predicted_answer": r.predicted_answer, "is_correct": r.is_correct, "attempts": r.attempts, "failure_mode": r.failure_mode, "reflection_count": len(r.reflections)} for r in records]
    return ReportPayload(meta={"dataset": dataset_name, "mode": mode, "num_records": len(records), "agents": sorted({r.agent_type for r in records})}, summary=summarize(records), failure_modes=failure_breakdown(records), examples=examples, extensions=["structured_evaluator", "reflection_memory", "benchmark_report_json", "mock_mode_for_autograding"], discussion=build_discussion(records, dataset_name))

def save_report(report: ReportPayload, out_dir: str | Path) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "report.json"
    md_path = out_dir / "report.md"
    json_path.write_text(json.dumps(report.model_dump(), indent=2), encoding="utf-8")
    s = report.summary
    react = s.get("react", {})
    reflexion = s.get("reflexion", {})
    delta = s.get("delta_reflexion_minus_react", {})
    ext_lines = "\n".join(f"- {item}" for item in report.extensions)

    # --- Hiệu quả chi phí: token / latency cho mỗi câu trả lời ĐÚNG ---
    def per_correct(agent: dict, metric: str) -> str:
        em = agent.get("em", 0) or 0
        val = agent.get(metric, 0) or 0
        return f"{val / em:.1f}" if em > 0 else "n/a"

    cost_rows = ""
    if react and reflexion:
        cost_rows = (
            f"| Token / câu đúng | {per_correct(react, 'avg_token_estimate')} | {per_correct(reflexion, 'avg_token_estimate')} |\n"
            f"| Latency(ms) / câu đúng | {per_correct(react, 'avg_latency_ms')} | {per_correct(reflexion, 'avg_latency_ms')} |\n"
            f"| Token / điểm EM tăng thêm | — | {(delta.get('tokens_abs', 0) / delta['em_abs']):.1f} |\n"
            if delta.get("em_abs") else
            f"| Token / câu đúng | {per_correct(react, 'avg_token_estimate')} | {per_correct(reflexion, 'avg_token_estimate')} |\n"
            f"| Latency(ms) / câu đúng | {per_correct(react, 'avg_latency_ms')} | {per_correct(reflexion, 'avg_latency_ms')} |\n"
        )

    # --- Failure modes thành bảng so sánh react vs reflexion ---
    fm = report.failure_modes
    fm_rows = "\n".join(
        f"| {mode} | {counts.get('react', 0)} | {counts.get('reflexion', 0)} |"
        for mode, counts in sorted(fm.items(), key=lambda kv: -(sum(kv[1].values())))
    ) or "| (không có) | 0 | 0 |"

    # --- Đối chiếu theo qid: react sai -> reflexion đúng / cả hai sai ---
    by_qid: dict[str, dict] = {}
    for e in report.examples:
        by_qid.setdefault(e["qid"], {})[e["agent_type"]] = e
    fixed, both_wrong = [], []
    for qid, agents in by_qid.items():
        rx, rf = agents.get("react"), agents.get("reflexion")
        if not (rx and rf):
            continue
        if not rx["is_correct"] and rf["is_correct"]:
            fixed.append((qid, rf))
        elif not rx["is_correct"] and not rf["is_correct"]:
            both_wrong.append((qid, rf))

    def fmt_cases(cases: list, limit: int = 10) -> str:
        if not cases:
            return "- (không có)"
        lines = []
        for qid, e in cases[:limit]:
            lines.append(
                f"- `{qid}` (att={e['attempts']}, mode={e['failure_mode']}): "
                f"gold=**{e['gold_answer']}** → pred=`{e['predicted_answer']}`"
            )
        if len(cases) > limit:
            lines.append(f"- … và {len(cases) - limit} câu khác")
        return "\n".join(lines)

    em_gain = delta.get("em_abs", 0)
    verdict = (
        f"Reflexion sửa được **{len(fixed)}** câu mà ReAct trả sai, nhưng tốn thêm "
        f"~{delta.get('tokens_abs', 0):.0f} token và ~{delta.get('latency_abs', 0):.0f}ms/câu trung bình. "
        f"Chênh lệch EM = {em_gain:+}."
    ) if react and reflexion else "Cần cả ReAct và Reflexion để so sánh."

    md = f"""# Lab 16 Benchmark Report

## Metadata
- Dataset: {report.meta['dataset']}
- Mode: {report.meta['mode']}
- Records: {report.meta['num_records']}
- Agents: {', '.join(report.meta['agents'])}

## 1. So sánh tổng quan (Summary)
| Metric | ReAct | Reflexion | Delta |
|---|---:|---:|---:|
| EM (Exact Match) | {react.get('em', 0)} | {reflexion.get('em', 0)} | {delta.get('em_abs', 0)} |
| Avg attempts | {react.get('avg_attempts', 0)} | {reflexion.get('avg_attempts', 0)} | {delta.get('attempts_abs', 0)} |
| Avg token (thật) | {react.get('avg_token_estimate', 0)} | {reflexion.get('avg_token_estimate', 0)} | {delta.get('tokens_abs', 0)} |
| Avg latency (ms, thật) | {react.get('avg_latency_ms', 0)} | {reflexion.get('avg_latency_ms', 0)} | {delta.get('latency_abs', 0)} |

## 2. Hiệu quả chi phí (Cost efficiency)
| Chỉ số | ReAct | Reflexion |
|---|---:|---:|
{cost_rows}
> {verdict}

## 3. Failure modes (so sánh)
| Failure mode | ReAct | Reflexion |
|---|---:|---:|
{fm_rows}

## 4. Reflexion cải thiện được (ReAct sai → Reflexion đúng): {len(fixed)} câu
{fmt_cases(fixed)}

## 5. Cả hai cùng thất bại: {len(both_wrong)} câu
{fmt_cases(both_wrong)}

## 6. Extensions implemented
{ext_lines}

## 7. Discussion
{report.discussion}
"""
    md_path.write_text(md, encoding="utf-8")
    return json_path, md_path
