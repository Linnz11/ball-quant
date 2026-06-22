from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

from ball_quant.models import Combo, MatchAnalysis, Selection


OUTCOME_LABELS = {
    "home": "主胜/让胜",
    "draw": "平/让平",
    "away": "主负/让负",
}


def render_markdown_report(
    date: str,
    budget: float,
    analyses: List[MatchAnalysis],
    allocated: List[Combo],
    combo_groups: Dict[str, List[Combo]],
) -> str:
    lines: List[str] = []
    lines.append(f"# 每日竞彩盘口研究报告 - {date}")
    lines.append("")
    lines.append(f"- 预算：{budget:.0f} 元")
    lines.append(f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("- 方法：先用 Polymarket 找真实概率路径，再用体彩 SP 判断赔率是否值得，再用球队事实解释，最后用组合概率和仓位算法决定怎么买。")
    lines.append("- 风险提示：报告不是收益承诺；若 Polymarket、体彩 SP 或球队事实数据缺失，置信度会下降。")
    lines.append("")

    lines.append("## 1. 每场比赛摘要")
    for analysis in analyses:
        lines.extend(render_match_summary(analysis))
    lines.append("")

    lines.append("## 2. 玩法映射表")
    lines.extend(render_mapping_table(analyses))
    lines.append("")

    lines.append("## 3. 组合表")
    lines.extend(render_combo_table(allocated))
    lines.append("")

    lines.append("## 4. 最好情况")
    if allocated:
        best = max(allocated, key=lambda combo: combo.payout)
        lines.append(f"- 最高单票回款约 {best.payout:.2f} 元，净利润约 {best.profit:.2f} 元。")
        lines.append(f"- 对应路径：{best.selection_text}")
    else:
        lines.append("- 无合格组合，最好情况是选择不下注，保留本金。")
    lines.append("")

    lines.append("## 5. 最坏情况")
    total_stake = sum(combo.stake for combo in allocated)
    lines.append(f"- 若所有出票组合失败，最大损失为已分配仓位 {total_stake:.2f} 元。")
    if total_stake < budget:
        lines.append(f"- 未分配资金 {budget - total_stake:.2f} 元保留，不强行下注。")
    lines.append("")

    lines.append("## 6. 主路径")
    main = sorted(all_selections(analyses), key=lambda s: (s.probability, s.edge), reverse=True)[:5]
    if main:
        for selection in main:
            lines.append(
                f"- {selection.match_id} {selection.home}vs{selection.away} {selection.play}:{selection.outcome}，"
                f"条件：{selection.condition}，概率 {pct(selection.probability)}，SP {selection.sp:.2f}，edge {selection.edge:.2%}。"
            )
    else:
        lines.append("- 当前数据不足，无法形成主路径。")
    lines.append("")

    lines.append("## 7. 失败路径")
    for analysis in analyses:
        risky = [s for s in analysis.selections if s.edge < -0.12 or "exact_margin" in s.tags]
        for selection in risky[:3]:
            lines.append(
                f"- {analysis.match.match_id} {selection.play}:{selection.outcome} 风险：{selection.risk_label}，"
                f"条件 {selection.condition}，概率 {pct(selection.probability)}。"
            )
    if not any(s.edge < -0.12 or "exact_margin" in s.tags for s in all_selections(analyses)):
        lines.append("- 暂无特别突出的失败分支；主要风险来自市场概率误差和临场阵容变化。")
    lines.append("")

    lines.append("## 8. 体彩店口播")
    lines.extend(render_shop_script(allocated))
    lines.append("")

    deleted = combo_groups.get("deleted", [])
    if deleted:
        lines.append("## 删除组合记录")
        for combo in deleted[:10]:
            lines.append(f"- 删除：{combo.selection_text}；原因：{combo.deletion_reason}")
        lines.append("")

    return "\n".join(lines)


def render_match_summary(analysis: MatchAnalysis) -> List[str]:
    match = analysis.match
    avg_spread, liquidity = analysis.matrix.liquidity_snapshot()
    selections = sorted(analysis.selections, key=lambda s: (s.probability, s.edge), reverse=True)
    main_path = selections[0] if selections else None
    lines = [
        f"### {match.match_id} {match.home} vs {match.away}",
        f"- 体彩 SP：胜 {match.spf_home:.2f} / 平 {match.spf_draw:.2f} / 负 {match.spf_away:.2f}；让球 {match.handicap:+d}，让胜 {match.rq_home:.2f} / 让平 {match.rq_draw:.2f} / 让负 {match.rq_away:.2f}",
        f"- Polymarket：event `{analysis.matrix.event_slug or '未匹配'}`；市场数 {len(analysis.matrix.markets)}；平均 spread {fmt_optional(avg_spread)}；流动性 {fmt_optional(liquidity)}",
        f"- 球队事实：{analysis.facts.home_summary}；{analysis.facts.away_summary}",
    ]
    if analysis.facts.warnings:
        lines.append(f"- 数据警告：{'；'.join(analysis.facts.warnings)}")
    if analysis.deleted_paths:
        lines.append(f"- 数据/赔率缺口：{'；'.join(analysis.deleted_paths[:3])}")
    if main_path:
        lines.append(
            f"- 主路径：{main_path.play}:{main_path.outcome}，{main_path.condition}，概率 {pct(main_path.probability)}，edge {main_path.edge:.2%}，{main_path.risk_label}"
        )
    deleted = [s for s in selections if s.edge < -0.12]
    if deleted:
        lines.append(f"- 删除路径：{'; '.join(f'{s.play}:{s.outcome}({s.risk_label})' for s in deleted[:3])}")
    else:
        lines.append("- 删除路径：暂无明显赔率不足路径，仍需看临场阵容和 SP 变化。")
    return lines + [""]


def render_mapping_table(analyses: List[MatchAnalysis]) -> List[str]:
    lines = [
        "| 比赛 | 体彩玩法 | 实际比分条件 | 概率 | SP | fair odds | edge | 是否保留 |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    for analysis in analyses:
        for selection in analysis.selections:
            keep = "保留" if selection.edge >= -0.08 and selection.confidence >= 0.35 else "删除/观察"
            lines.append(
                f"| {selection.match_id} {selection.home}vs{selection.away} | {selection.play}:{selection.outcome} | "
                f"{selection.condition} | {pct(selection.probability)} | {selection.sp:.2f} | "
                f"{selection.fair_odds:.2f} | {selection.edge:.2%} | {keep} |"
            )
    return lines


def render_combo_table(combos: List[Combo]) -> List[str]:
    lines = [
        "| 组合 | 概率 | 赔率 | EV | Kelly | 金额 | 回款 | 净利润 | 类型 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    if not combos:
        lines.append("| 无合格组合 | - | - | - | - | 0 | 0 | 0 | 不下注 |")
        return lines
    for combo in combos:
        lines.append(
            f"| {combo.selection_text} | {pct(combo.probability)} | {combo.odds:.2f} | "
            f"{combo.expected_return:.2%} | {combo.kelly:.2%} | "
            f"{combo.stake:.2f} | {combo.payout:.2f} | {combo.profit:.2f} | {combo.combo_type} |"
        )
    return lines


def render_shop_script(combos: List[Combo]) -> List[str]:
    if not combos:
        return ["- 今天没有达到阈值的主票，建议不出票或只观察临场。"]
    lines = []
    for combo in combos:
        parts = []
        for selection in combo.selections:
            label = shop_outcome(selection)
            parts.append(f"{selection.match_id} {label}")
        lines.append(f"- {combo.combo_type}：竞彩足球，{ '，'.join(parts) }，{len(combo.selections)} 串 1，金额 {combo.stake:.0f} 元。")
    return lines


def shop_outcome(selection: Selection) -> str:
    if selection.play == "spf":
        return {"home": "主胜", "draw": "平", "away": "主负"}[selection.outcome]
    return {"home": "让胜", "draw": "让平", "away": "让负"}[selection.outcome]


def all_selections(analyses: Iterable[MatchAnalysis]) -> List[Selection]:
    selections: List[Selection] = []
    for analysis in analyses:
        selections.extend(analysis.selections)
    return selections


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def fmt_optional(value) -> str:
    return "-" if value is None else f"{value:.4g}"


def write_report(path: str, content: str) -> Path:
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(content, encoding="utf-8")
    return report_path
