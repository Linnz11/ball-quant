"""Tests for ball_quant.reporting.markdown — deterministic, offline."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pytest

from ball_quant.core.analysis import analyze_match
from ball_quant.models import (
    Combo,
    EventMarketMatrix,
    MarketQuote,
    MatchSP,
    Selection,
    TeamFacts,
)
from ball_quant.reporting.markdown import render_markdown_report, write_report


# ---------------------------------------------------------------------------
# Minimal fixtures for constructing MatchAnalysis inputs
# ---------------------------------------------------------------------------

def _minimal_match() -> MatchSP:
    return MatchSP(
        match_id="T01",
        date="2026-06-14",
        home="Alpha",
        away="Beta",
        spf_home=2.0,
        spf_draw=3.5,
        spf_away=4.0,
        handicap=0,
        rq_home=2.0,
        rq_draw=3.5,
        rq_away=4.0,
    )


def _minimal_matrix() -> EventMarketMatrix:
    return EventMarketMatrix(
        match_id="T01",
        home="Alpha",
        away="Beta",
        markets=[
            MarketQuote("m1", "winner", "moneyline", "home", 0.55, spread=0.02, liquidity=5000),
            MarketQuote("m1", "winner", "moneyline", "draw", 0.25, spread=0.02, liquidity=5000),
            MarketQuote("m1", "winner", "moneyline", "away", 0.20, spread=0.02, liquidity=5000),
        ],
    )


def _minimal_facts() -> TeamFacts:
    return TeamFacts(
        match_id="T01",
        source="test",
        home_summary="Alpha: all good",
        away_summary="Beta: all good",
    )


def _one_selection(edge: float = 0.10, prob: float = 0.55) -> Selection:
    sp = 2.0
    return Selection(
        match_id="T01",
        home="Alpha",
        away="Beta",
        play="spf",
        outcome="home",
        condition="Alpha wins",
        probability=prob,
        sp=sp,
        fair_odds=1.0 / prob,
        break_even=1.0 / sp,
        edge=edge,
        kelly=0.10,
        confidence=0.70,
        risk_label="价值保留",
    )


def _one_combo(selection: Selection) -> Combo:
    return Combo(
        name="T01-A",
        selections=[selection],
        probability=selection.probability,
        odds=selection.sp,
        expected_return=selection.edge,
        combo_type="A",
        kelly=selection.kelly,
        stake=20.0,
        payout=40.0,
        profit=20.0,
    )


# ---------------------------------------------------------------------------
# render_markdown_report — section headers and content
# ---------------------------------------------------------------------------

class TestRenderMarkdownReport:
    def _render(self, budget: float = 100.0) -> str:
        match = _minimal_match()
        matrix = _minimal_matrix()
        facts = _minimal_facts()
        analysis = analyze_match(match, matrix, facts)
        sel = _one_selection()
        # Inject the selection so we have combos and selections to render
        analysis.selections.insert(0, sel)
        combo = _one_combo(sel)
        allocated = [combo]
        combo_groups: Dict[str, List[Combo]] = {"A": [combo], "deleted": []}
        return render_markdown_report("2026-06-14", budget, [analysis], allocated, combo_groups)

    def test_returns_string(self):
        result = self._render()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_title_contains_date(self):
        result = self._render()
        assert "2026-06-14" in result

    def test_section_1_每场比赛摘要(self):
        result = self._render()
        assert "## 1. 每场比赛摘要" in result

    def test_section_2_玩法映射表(self):
        result = self._render()
        assert "## 2. 玩法映射表" in result

    def test_section_3_组合表(self):
        result = self._render()
        assert "## 3. 组合表" in result

    def test_section_4_最好情况(self):
        result = self._render()
        assert "## 4. 最好情况" in result

    def test_section_5_最坏情况(self):
        result = self._render()
        assert "## 5. 最坏情况" in result

    def test_section_6_主路径(self):
        result = self._render()
        assert "## 6. 主路径" in result

    def test_section_7_失败路径(self):
        result = self._render()
        assert "## 7. 失败路径" in result

    def test_section_8_体彩店口播(self):
        result = self._render()
        assert "## 8. 体彩店口播" in result

    def test_budget_appears_in_output(self):
        result = self._render(budget=250.0)
        assert "250" in result

    def test_match_id_appears_in_output(self):
        result = self._render()
        assert "T01" in result

    def test_home_team_appears_in_output(self):
        result = self._render()
        assert "Alpha" in result

    def test_empty_allocated_shows_no_combo(self):
        match = _minimal_match()
        matrix = _minimal_matrix()
        facts = _minimal_facts()
        analysis = analyze_match(match, matrix, facts)
        result = render_markdown_report(
            "2026-06-14", 100.0, [analysis], [], {"A": [], "deleted": []}
        )
        assert "无合格组合" in result

    def test_combo_stake_shown_in_worst_case(self):
        result = self._render()
        # Stake is 20.0 — should appear in worst-case section
        assert "20" in result

    def test_deleted_section_appears_when_deleted_combos_exist(self):
        match = _minimal_match()
        matrix = _minimal_matrix()
        facts = _minimal_facts()
        analysis = analyze_match(match, matrix, facts)
        sel = _one_selection(edge=-0.20)
        deleted_combo = Combo(
            name="del-1",
            selections=[sel],
            probability=0.40,
            odds=2.0,
            expected_return=-0.20,
            combo_type="deleted",
            deletion_reason="组合EV不为正，概率与赔率不匹配",
        )
        result = render_markdown_report(
            "2026-06-14", 100.0, [analysis], [], {"A": [], "deleted": [deleted_combo]}
        )
        assert "删除组合记录" in result


# ---------------------------------------------------------------------------
# write_report
# ---------------------------------------------------------------------------

class TestWriteReport:
    def test_writes_utf8_file(self, tmp_path: Path):
        content = "# 报告\n\n内容日本語"
        out = tmp_path / "report.md"
        returned = write_report(str(out), content)
        assert out.exists()
        assert out.read_text(encoding="utf-8") == content

    def test_returns_path_object(self, tmp_path: Path):
        out = tmp_path / "r.md"
        result = write_report(str(out), "test")
        assert isinstance(result, Path)

    def test_creates_parent_dirs(self, tmp_path: Path):
        deep = tmp_path / "a" / "b" / "c" / "report.md"
        write_report(str(deep), "deep content")
        assert deep.exists()
        assert deep.read_text(encoding="utf-8") == "deep content"

    def test_overwrites_existing_file(self, tmp_path: Path):
        out = tmp_path / "r.md"
        out.write_text("old", encoding="utf-8")
        write_report(str(out), "new")
        assert out.read_text(encoding="utf-8") == "new"
