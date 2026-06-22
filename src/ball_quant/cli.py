from __future__ import annotations

import argparse
import csv
import json
import time
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from ball_quant.adapters.api_football import APIFootballClient
from ball_quant.adapters.polymarket import (
    WORLD_CUP_TAG_ID,
    PolymarketClient,
    event_to_quotes,
    flatten_inventory,
    infer_match_teams,
    load_matrices_from_file,
    matrix_to_inventory,
)
from ball_quant.adapters.ticai import load_ticai_matches
from ball_quant.core.analysis import analyze_match, flatten_selections
from ball_quant.core.causal import causal_layer_summary
from ball_quant.core.combo import generate_combos
from ball_quant.core.handicap import handicap_condition
from ball_quant.core.probability import build_probability_context, probability_for_handicap, probability_for_spf
from ball_quant.core.schedule import select_schedule_rows
from ball_quant.core.snapshot import build_live_probability_snapshot
from ball_quant.core.staking import allocate_stakes
from ball_quant.diagnostics import polymarket_doctor
from ball_quant.models import EventMarketMatrix, MatchSP, MarketQuote
from ball_quant.reporting.markdown import render_markdown_report, write_report
from ball_quant.config.settings import Settings
from ball_quant.logging_setup import configure_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ballq",
        description="Daily football betting market research system.",
    )
    sub = parser.add_subparsers(dest="command")
    doctor = sub.add_parser("doctor", help="Check proxy and Polymarket API connectivity")
    doctor.add_argument("--timeout", type=int, default=15, help="HTTP timeout seconds")
    dump = sub.add_parser("poly-dump", help="Fetch and export live Polymarket World Cup market inventory")
    dump.add_argument("--query", action="append", help="Polymarket search query; repeatable")
    dump.add_argument("--slug", action="append", help="Direct Polymarket event slug; repeatable")
    dump.add_argument("--slug-only", action="store_true", help="Only fetch explicit --slug values; skip default/search queries")
    dump.add_argument("--skip-world-cup-tag", action="store_true", help="Skip the default World Cup tag/keyset crawl")
    dump.add_argument("--world-cup-tag-id", type=int, default=WORLD_CUP_TAG_ID, help="Gamma tag id used for the World Cup matrix")
    dump.add_argument("--max-world-cup-events", type=int, default=700, help="Max events to fetch from the World Cup tag/keyset crawl")
    dump.add_argument("--with-sports-pages", action="store_true", help="Fetch Polymarket Sports pages for match slugs to expand props/spreads/totals")
    dump.add_argument("--limit-per-type", type=int, default=20, help="Search result limit per type")
    dump.add_argument("--include-closed", action="store_true", help="Include closed markets in search")
    dump.add_argument("--with-orderbook", action="store_true", help="Also fetch CLOB bid/ask orderbooks")
    dump.add_argument("--orderbook-limit", type=int, default=40, help="Max quotes to enrich per event when fetching orderbooks")
    dump.add_argument("--json-out", default="data/cache/polymarket_worldcup_inventory.json")
    dump.add_argument("--csv-out", default="data/cache/polymarket_worldcup_inventory.csv")
    poly_match = sub.add_parser("poly-match", help="Analyze one Polymarket Sports match into normalized probability branches")
    poly_match.add_argument("--slug", required=True, help="Polymarket event slug, for example fifwc-nld-jpn-2026-06-14")
    poly_match.add_argument("--out", default=None, help="Optional Markdown output path")
    poly_match.add_argument("--with-orderbook", action="store_true", help="Also fetch CLOB bid/ask orderbooks")
    poly_match.add_argument("--orderbook-limit", type=int, default=120, help="Max quotes to enrich when fetching orderbooks")
    schedule = sub.add_parser("poly-schedule", help="List/sync Polymarket World Cup core match schedule with timezone alignment")
    schedule.add_argument("--date", help="Date filter in YYYY-MM-DD; defaults to all active matches")
    schedule.add_argument("--date-mode", choices=("poly", "local", "all"), default="poly", help="Interpret --date as Polymarket eventDate or local date")
    schedule.add_argument("--timezone", default="Asia/Shanghai", help="IANA timezone used for local kickoff display")
    schedule.add_argument("--world-cup-tag-id", type=int, default=WORLD_CUP_TAG_ID)
    schedule.add_argument("--max-world-cup-events", type=int, default=700)
    schedule.add_argument("--include-closed", action="store_true", help="Also fetch closed events for schedule history")
    schedule.add_argument("--keep-expired", action="store_true", help="Keep expired/closed matches in outputs")
    schedule.add_argument("--lookahead-hours", type=float, help="Only include matches starting within this many hours")
    schedule.add_argument("--expire-after-hours", type=float, default=3.0, help="Hours after kickoff before a non-closed match is pruned")
    schedule.add_argument("--json-out", default="data/cache/poly_worldcup_active_schedule.json")
    schedule.add_argument("--csv-out", default="data/cache/poly_worldcup_active_schedule.csv")
    auto = sub.add_parser("auto-refresh", help="Refresh active Polymarket schedule, live match matrices, and Markdown reports")
    auto.add_argument("--timezone", default="Asia/Shanghai")
    auto.add_argument("--world-cup-tag-id", type=int, default=WORLD_CUP_TAG_ID)
    auto.add_argument("--max-world-cup-events", type=int, default=700)
    auto.add_argument("--lookahead-hours", type=float, default=36.0, help="Refresh matches starting within this many hours")
    auto.add_argument("--expire-after-hours", type=float, default=3.0)
    auto.add_argument("--with-orderbook", action="store_true", help="Also fetch CLOB bid/ask orderbooks for refreshed match matrices")
    auto.add_argument("--orderbook-limit", type=int, default=120)
    auto.add_argument("--schedule-json-out", default="data/cache/poly_worldcup_active_schedule.json")
    auto.add_argument("--schedule-csv-out", default="data/cache/poly_worldcup_active_schedule.csv")
    auto.add_argument("--live-cache-dir", default="data/cache/live")
    auto.add_argument("--reports-dir", default="reports/live")
    auto.add_argument("--skip-reports", action="store_true", help="Only write the active schedule; skip per-match matrices/reports")
    auto.add_argument("--keep-expired-files", action="store_true", help="Do not remove old files from live cache/report directories")
    run = sub.add_parser("run", help="Generate daily Markdown research report")
    run.add_argument("--date", required=True, help="Match date, YYYY-MM-DD")
    run.add_argument("--budget", required=True, type=float, help="Total staking budget in RMB")
    run.add_argument("--sp-file", required=True, help="China Sports Lottery SP CSV/HTML file")
    run.add_argument("--html-file", help="Optional extra HTML SP file")
    run.add_argument("--competition", default=None, help="Competition hint for Polymarket event matching")
    run.add_argument("--polymarket-cache", help="JSON fixture containing preloaded Polymarket matrices")
    run.add_argument("--offline-cache", action="store_true", help="Do not call external APIs; use fixtures/cache only")
    run.add_argument("--refresh-polymarket", action="store_true", help="Force fresh Polymarket API fetch even if cache exists")
    run.add_argument("--polymarket-cache-ttl", type=int, default=120, help="Seconds before cached Polymarket data is considered stale")
    run.add_argument("--report-out", help="Output report path")

    # ---- capture ----------------------------------------------------------------
    capture_p = sub.add_parser(
        "capture",
        help="Fetch a match's EventMarketMatrix and persist a bq.snapshot.v1 record",
    )
    capture_p.add_argument("--slug", required=True, help="Polymarket event slug, e.g. fifwc-nld-jpn-2026-06-14")
    capture_p.add_argument("--sp-file", default=None, help="China Sports Lottery SP CSV for MatchSP odds (optional)")
    capture_p.add_argument("--competition", default=None, help="Competition hint for slug league detection")
    capture_p.add_argument("--polymarket-cache", default=None, help="Offline JSON fixture; skip live API fetch")
    capture_p.add_argument("--store-root", default=None, help="Snapshot store root (default: settings.store_root)")
    capture_p.add_argument("--with-orderbook", action="store_true", help="Enrich with CLOB bid/ask")
    capture_p.add_argument("--orderbook-limit", type=int, default=120)

    # ---- settle -----------------------------------------------------------------
    settle_p = sub.add_parser(
        "settle",
        help="Load final match results CSV and persist outcomes JSON to the store",
    )
    settle_p.add_argument("--results", required=True, help="CSV path: match_id,home_score,away_score[,void]")
    settle_p.add_argument("--store-root", default=None, help="Snapshot store root (default: settings.store_root)")
    settle_p.add_argument("--out", default=None, help="Override output JSON path (default: <store>/outcomes/results.json)")

    # ---- backtest ---------------------------------------------------------------
    backtest_p = sub.add_parser(
        "backtest",
        help="Replay snapshots in a date range and produce a calibration/PnL report",
    )
    backtest_p.add_argument("--from", dest="date_from", required=True, help="Start date YYYY-MM-DD (inclusive)")
    backtest_p.add_argument("--to", dest="date_to", required=True, help="End date YYYY-MM-DD (inclusive)")
    backtest_p.add_argument("--match", default=None, help="Filter to a single match_id")
    backtest_p.add_argument("--params", default=None, help="JSON file or inline JSON with StrategyParams overrides")
    backtest_p.add_argument("--budget", type=float, default=None, help="Per-match budget (default: settings.default_budget)")
    backtest_p.add_argument("--bankroll", type=float, default=None, help="Total bankroll (default: settings.default_bankroll)")
    backtest_p.add_argument("--store-root", default=None, help="Snapshot store root")
    backtest_p.add_argument("--results", default=None, help="Outcomes CSV or JSON (previously saved by settle)")
    backtest_p.add_argument("--report-out", default=None, help="Markdown output path (default: reports/backtest_<from>_<to>.md)")
    backtest_p.add_argument("--profiles", default=None, help="ParamProfiles JSON (from `optimize --by-competition`); enables per-competition params")

    # ---- optimize ---------------------------------------------------------------
    optimize_p = sub.add_parser(
        "optimize",
        help="Walk-forward parameter search over a snapshot date range",
    )
    optimize_p.add_argument("--space", required=True, help="JSON: {field:[v,...]} for grid or {field:[lo,hi]} for random")
    optimize_p.add_argument("--metric", default="brier", help="Optimisation metric (brier/log_loss/ece/net_pnl/roi/geometric_growth_rate)")
    optimize_p.add_argument("--search", choices=("grid", "random"), default="grid")
    optimize_p.add_argument("--folds", type=int, default=3, dest="n_folds")
    optimize_p.add_argument("--from", dest="date_from", required=True, help="Start date YYYY-MM-DD")
    optimize_p.add_argument("--to", dest="date_to", required=True, help="End date YYYY-MM-DD")
    optimize_p.add_argument("--max-trials", type=int, default=None, help="Required for random search")
    optimize_p.add_argument("--seed", type=int, default=0)
    optimize_p.add_argument("--budget", type=float, default=None)
    optimize_p.add_argument("--bankroll", type=float, default=None)
    optimize_p.add_argument("--store-root", default=None)
    optimize_p.add_argument("--results", default=None, help="Outcomes CSV or JSON")
    optimize_p.add_argument("--report-out", default=None, help="Markdown output path")
    optimize_p.add_argument("--by-competition", action="store_true", help="Run per-competition optimization and emit a ParamProfiles JSON")
    optimize_p.add_argument("--profiles-out", default=None, help="Path to write ParamProfiles JSON (only with --by-competition)")

    # ---- recommend --------------------------------------------------------------
    rec = sub.add_parser(
        "recommend",
        help="Generate a turnkey 体彩 betting slip for today's budget",
    )
    rec.add_argument("--budget", required=True, type=float, help="Total staking budget in RMB (required)")
    rec.add_argument("--bankroll", type=float, default=None, help="Total bankroll (default: budget)")
    rec.add_argument("--date", default=None, help="Match date YYYY-MM-DD (default: all dates in feed)")
    # 体彩 source — exactly one required (validated at runtime)
    rec_src = rec.add_mutually_exclusive_group(required=True)
    rec_src.add_argument(
        "--sporttery-cache",
        dest="sporttery_cache",
        help="Path to a saved getMatchCalculatorV1 JSON payload (offline cassette)",
    )
    rec_src.add_argument(
        "--sp-file",
        dest="sp_file",
        help="Existing ticai CSV — parsed via adapters.ticai.load_ticai_matches (spf/handicap only)",
    )
    rec_src.add_argument(
        "--live-sporttery",
        dest="live_sporttery",
        action="store_true",
        help="Fetch fresh odds from sporttery API (requires China-IP; will 567 outside China)",
    )
    rec_src.add_argument(
        "--c500-cache",
        dest="c500_cache",
        default=None,
        help="Directory containing c500_*.html cassette files for offline 500.com replay",
    )
    rec_src.add_argument(
        "--c500-live",
        dest="c500_live",
        action="store_true",
        help="Fetch fresh 竞彩 odds from trade.500.com (requires network access to 500.com)",
    )
    # Polymarket source
    rec.add_argument(
        "--polymarket-cache",
        dest="polymarket_cache",
        default=None,
        help="JSON fixture with Polymarket matrices — file with {matrices:[...]} or legacy {matches:{...}}",
    )
    # Strategy knobs
    rec.add_argument("--min-edge", dest="min_edge", type=float, default=0.0, help="Minimum EV edge gate (default: 0.0)")
    rec.add_argument(
        "--max-legs",
        dest="max_legs",
        type=int,
        default=4,
        help="Maximum legs in a parlay combo (default: 4; 竞彩 串关 fidelity guard)",
    )
    rec.add_argument("--params", default=None, help="JSON file or inline JSON StrategyParams overrides")
    rec.add_argument(
        "--report-out",
        dest="report_out",
        default=None,
        help="Markdown slip output path (default: reports/recommend_<date>.md)",
    )
    rec.add_argument(
        "--json-out",
        dest="json_out",
        default=None,
        help="JSON slip output path for LLM consumption (default: reports/recommend_<date>.json)",
    )

    # ---- kg-build -----------------------------------------------------------
    kg = sub.add_parser(
        "kg-build",
        help="Build / refresh the team knowledge graph from Polymarket futures markets",
    )
    kg_src = kg.add_mutually_exclusive_group(required=True)
    kg_src.add_argument(
        "--poly-dump",
        dest="poly_dump",
        default=None,
        help="Path to a saved poly-dump JSON file (offline; produced by `ballq poly-dump`)",
    )
    kg_src.add_argument(
        "--live",
        action="store_true",
        help="Fetch live World Cup events from Polymarket and extract futures quotes",
    )
    kg.add_argument(
        "--devig",
        choices=("proportional", "shin"),
        default="proportional",
        help="Devig method for multi-outcome futures markets (default: proportional)",
    )
    kg.add_argument(
        "--kg-out",
        dest="kg_out",
        default="data/kg/teams.json",
        help="Output path for the team knowledge graph JSON (default: data/kg/teams.json)",
    )
    kg.add_argument(
        "--world-cup-tag-id",
        type=int,
        default=WORLD_CUP_TAG_ID,
        help="Gamma tag id for live WC event fetch (only used with --live)",
    )
    kg.add_argument(
        "--max-world-cup-events",
        type=int,
        default=700,
        help="Max WC events to fetch from Polymarket tag crawl (only used with --live)",
    )

    # ---- bundle -------------------------------------------------------------
    bnd = sub.add_parser(
        "bundle",
        help="Emit per-match RAW data bundle (Poly all-angles + 体彩 + flags + KG) for the LLM — no edge, no slip",
    )
    bnd.add_argument("--date", required=True, help="Match date YYYY-MM-DD")
    bnd_src = bnd.add_mutually_exclusive_group(required=True)
    bnd_src.add_argument(
        "--c500-live", dest="c500_live", action="store_true",
        help="Fetch fresh 竞彩 odds from trade.500.com",
    )
    bnd_src.add_argument(
        "--c500-cache", dest="c500_cache", default=None,
        help="Directory with c500_*.html cassettes for offline replay",
    )
    bnd.add_argument("--kg", dest="kg_path", default=None, help="KG teams.json path (default: data/kg/teams.json)")
    bnd.add_argument(
        "--out", dest="out", default=None,
        help="Output basename; writes <out>.json + <out>.md (default reports/bundle_<date>)",
    )
    bnd.add_argument(
        "--forecast-ledger", dest="forecast_ledger", default=None,
        help="Path for forecast ledger JSONL (default: data/forecasts/ledger.jsonl)",
    )

    grd = sub.add_parser(
        "grade",
        help="Grade forecast ledger vs results → calibration table (Brier/log-loss/ECE per forecaster×market)",
    )
    grd.add_argument(
        "--results", required=True,
        help="CSV of actual scores: match_id,home_score,away_score[,void]",
    )
    grd.add_argument(
        "--ledger", default=None,
        help="Forecast ledger JSONL path (default: data/forecasts/ledger.jsonl)",
    )
    grd.add_argument(
        "--date", default=None,
        help="Filter ledger by match_date YYYY-MM-DD (default: all dates, accumulates)",
    )
    grd.add_argument(
        "--out", default=None,
        help="Write calibration report to this file (default: print only)",
    )

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Configure logging as early as possible so all subsequent code has a logger.
    settings = Settings.load()
    configure_logging(settings.log_level)

    if args.command == "doctor":
        return doctor(args)
    if args.command == "poly-dump":
        return poly_dump(args)
    if args.command == "poly-match":
        return poly_match(args)
    if args.command == "poly-schedule":
        return poly_schedule(args)
    if args.command == "bundle":
        return cmd_bundle(args, settings)
    if args.command == "auto-refresh":
        return auto_refresh(args)
    if args.command == "run":
        return run(args)
    if args.command == "capture":
        return cmd_capture(args, settings)
    if args.command == "settle":
        return cmd_settle(args, settings)
    if args.command == "backtest":
        return cmd_backtest(args, settings)
    if args.command == "optimize":
        return cmd_optimize(args, settings)
    if args.command == "recommend":
        return cmd_recommend(args, settings)
    if args.command == "kg-build":
        return cmd_kg_build(args)
    if args.command == "grade":
        return cmd_grade(args, settings)
    parser.print_help()
    return 1


def doctor(args: argparse.Namespace) -> int:
    result = polymarket_doctor(timeout=args.timeout)
    print("Proxy:", result["proxies"] or "none")
    for key in ("gamma", "clob"):
        item = result[key]
        if item["ok"]:
            print(f"{key}: OK {item['seconds']}s {item['summary']}")
        else:
            print(f"{key}: FAIL {item['seconds']}s {item['error']}")
    return 0 if result["gamma"]["ok"] and result["clob"]["ok"] else 2


def poly_dump(args: argparse.Namespace) -> int:
    use_world_cup_tag = not args.slug_only and not args.skip_world_cup_tag
    queries = [] if args.slug_only else (args.query or ([] if use_world_cup_tag else default_world_cup_queries()))
    client = PolymarketClient(
        cache_dir=Path("data/cache"),
        refresh=True,
        enrich_orderbook=args.with_orderbook,
        enrich_sports_payload=args.with_sports_pages or bool(args.slug),
    )
    events = []
    seen = set()
    for slug in args.slug or []:
        event = client.get_event_by_slug(slug)
        key = event.get("slug") or event.get("id")
        if key and key not in seen:
            seen.add(key)
            events.append(event)
    if use_world_cup_tag:
        for event in client.fetch_world_cup_events(
            tag_id=args.world_cup_tag_id,
            max_events=args.max_world_cup_events,
            include_closed=args.include_closed,
        ):
            key = event.get("slug") or event.get("id")
            if key and key not in seen:
                seen.add(key)
                events.append(event)
    if queries:
        for event in client.search_world_cup_events(
            queries=queries,
            limit_per_type=args.limit_per_type,
            include_closed=args.include_closed,
        ):
            key = event.get("slug") or event.get("id")
            if key and key not in seen:
                seen.add(key)
                events.append(event)
    inventories = [
        client.event_inventory(
            event,
            enrich_orderbook=args.with_orderbook,
            orderbook_limit=args.orderbook_limit,
        )
        for event in events
    ]
    payload = {
        "fetched_at": time.time(),
        "queries": queries,
        "world_cup_tag_id": args.world_cup_tag_id if use_world_cup_tag else None,
        "max_world_cup_events": args.max_world_cup_events if use_world_cup_tag else None,
        "sports_pages": bool(args.with_sports_pages or args.slug),
        "event_count": len(inventories),
        "quote_count": sum(item["quote_count"] for item in inventories),
        "events": inventories,
    }
    json_path = Path(args.json_out)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_inventory_csv(args.csv_out, flatten_inventory(inventories))
    print(f"Polymarket inventory: events={payload['event_count']} quotes={payload['quote_count']}")
    print(f"JSON written: {json_path}")
    print(f"CSV written: {Path(args.csv_out)}")
    return 0


def poly_match(args: argparse.Namespace) -> int:
    client = PolymarketClient(
        cache_dir=Path("data/cache"),
        refresh=True,
        enrich_orderbook=args.with_orderbook,
        enrich_sports_payload=True,
    )
    event = client.prefer_sports_event(client.get_event_by_slug(args.slug))
    title = str(event.get("title") or args.slug)
    home, away = infer_match_teams(title)
    matrix = EventMarketMatrix(
        match_id=str(event.get("id") or event.get("slug") or args.slug),
        home=home,
        away=away,
        event_id=str(event.get("id") or ""),
        event_slug=event.get("slug") or args.slug,
        markets=event_to_quotes(event, home, away),
        raw_event=event,
    )
    if args.with_orderbook:
        client.enrich_with_clob(matrix, max_quotes=args.orderbook_limit)
    report = render_poly_match_report(matrix)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report, encoding="utf-8")
        print(f"Polymarket match report written: {out}")
    else:
        print(report)
    return 0


def poly_schedule(args: argparse.Namespace) -> int:
    client = PolymarketClient(
        cache_dir=Path("data/cache"),
        refresh=True,
        enrich_orderbook=False,
        enrich_sports_payload=False,
    )
    events = fetch_world_cup_schedule_events(
        client,
        tag_id=args.world_cup_tag_id,
        max_events=args.max_world_cup_events,
        include_closed=args.include_closed,
    )
    date = None if args.date_mode == "all" else args.date
    rows = select_schedule_rows(
        events,
        timezone_name=args.timezone,
        date=date,
        date_mode=args.date_mode,
        include_expired=args.keep_expired,
        lookahead_hours=args.lookahead_hours,
        expire_after_hours=args.expire_after_hours,
    )
    write_schedule_outputs(
        rows,
        json_path=args.json_out,
        csv_path=args.csv_out,
        metadata={
            "fetched_at": time.time(),
            "timezone": args.timezone,
            "date": args.date,
            "date_mode": args.date_mode,
            "include_closed": args.include_closed,
            "keep_expired": args.keep_expired,
            "lookahead_hours": args.lookahead_hours,
        },
    )
    print_schedule(rows, args.timezone)
    print(f"JSON written: {Path(args.json_out)}")
    print(f"CSV written: {Path(args.csv_out)}")
    return 0


def auto_refresh(args: argparse.Namespace) -> int:
    client = PolymarketClient(
        cache_dir=Path("data/cache"),
        refresh=True,
        enrich_orderbook=args.with_orderbook,
        enrich_sports_payload=True,
    )
    events = fetch_world_cup_schedule_events(
        client,
        tag_id=args.world_cup_tag_id,
        max_events=args.max_world_cup_events,
        include_closed=False,
    )
    rows = select_schedule_rows(
        events,
        timezone_name=args.timezone,
        include_expired=False,
        lookahead_hours=args.lookahead_hours,
        expire_after_hours=args.expire_after_hours,
    )
    write_schedule_outputs(
        rows,
        json_path=args.schedule_json_out,
        csv_path=args.schedule_csv_out,
        metadata={
            "fetched_at": time.time(),
            "timezone": args.timezone,
            "lookahead_hours": args.lookahead_hours,
            "expire_after_hours": args.expire_after_hours,
        },
    )
    refreshed_reports = []
    probability_snapshots = []
    active_slugs = {row["event_slug"] for row in rows if row.get("event_slug")}
    schedule_by_slug = {row["event_slug"]: row for row in rows if row.get("event_slug")}
    live_cache_dir = Path(args.live_cache_dir)
    reports_dir = Path(args.reports_dir)
    if not args.skip_reports:
        live_cache_dir.mkdir(parents=True, exist_ok=True)
        reports_dir.mkdir(parents=True, exist_ok=True)
        for row in rows:
            slug = row.get("event_slug")
            if not slug:
                continue
            event = client.prefer_sports_event(client.get_event_by_slug(str(slug)))
            title = str(event.get("title") or slug)
            home, away = infer_match_teams(title)
            matrix = EventMarketMatrix(
                match_id=str(event.get("id") or event.get("slug") or slug),
                home=home,
                away=away,
                event_id=str(event.get("id") or ""),
                event_slug=event.get("slug") or str(slug),
                markets=event_to_quotes(event, home, away),
                raw_event=event,
            )
            if args.with_orderbook:
                client.enrich_with_clob(matrix, max_quotes=args.orderbook_limit)
            inventory = matrix_to_inventory(matrix)
            matrix_json = live_cache_dir / f"{slug}.json"
            matrix_json.write_text(json.dumps(inventory, ensure_ascii=False, indent=2), encoding="utf-8")
            write_inventory_csv(str(live_cache_dir / f"{slug}.csv"), flatten_inventory([inventory]))
            snapshot = build_live_probability_snapshot(matrix, local_schedule=schedule_by_slug.get(slug))
            probability_snapshots.append(snapshot)
            snapshot_path = live_cache_dir / f"{slug}_probability.json"
            snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
            report_path = reports_dir / f"poly_match_{slug}.md"
            report_path.write_text(render_poly_match_report(matrix), encoding="utf-8")
            refreshed_reports.append(str(report_path))
        if not args.keep_expired_files:
            prune_live_files(live_cache_dir, reports_dir, active_slugs)
        write_probability_snapshot_outputs(live_cache_dir, probability_snapshots)
    status_path = Path("data/cache/poly_auto_refresh_status.json")
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(
        json.dumps(
            {
                "refreshed_at": time.time(),
                "timezone": args.timezone,
                "lookahead_hours": args.lookahead_hours,
                "active_match_count": len(rows),
                "refreshed_report_count": len(refreshed_reports),
                "probability_snapshot_count": len(probability_snapshots),
                "active_slugs": sorted(active_slugs),
                "reports": refreshed_reports,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Active matches: {len(rows)}")
    print(f"Reports refreshed: {len(refreshed_reports)}")
    print(f"Probability snapshots: {len(probability_snapshots)}")
    print(f"Schedule JSON: {Path(args.schedule_json_out)}")
    print(f"Status JSON: {status_path}")
    return 0


def fetch_world_cup_schedule_events(
    client: PolymarketClient,
    tag_id: int,
    max_events: int,
    include_closed: bool,
) -> List[Dict]:
    events = client.fetch_world_cup_events(
        tag_id=tag_id,
        max_events=max_events,
        include_closed=False,
    )
    if include_closed:
        events.extend(
            client.fetch_world_cup_events(
                tag_id=tag_id,
                max_events=max_events,
                include_closed=True,
            )
        )
    deduped = []
    seen = set()
    for event in events:
        key = event.get("slug") or event.get("id")
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(event)
    return deduped


def write_schedule_outputs(rows: List[Dict], json_path: str, csv_path: str, metadata: Dict) -> None:
    payload = {**metadata, "match_count": len(rows), "matches": rows}
    out = Path(json_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_schedule_csv(csv_path, rows)


def write_schedule_csv(path: str, rows: List[Dict]) -> None:
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "polymarket_date",
        "local_date",
        "local_time",
        "local_timezone",
        "event_title",
        "home",
        "away",
        "status",
        "start_time_utc",
        "event_slug",
        "event_id",
        "active",
        "closed",
        "ended",
        "updated_at",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_probability_snapshot_outputs(cache_dir: Path, snapshots: List[Dict]) -> None:
    payload = {
        "fetched_at": time.time(),
        "match_count": len(snapshots),
        "snapshots": snapshots,
    }
    (cache_dir / "live_probability_snapshots.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_probability_summary_csv(cache_dir / "live_probability_summary.csv", snapshots)


def write_probability_summary_csv(path: Path, snapshots: List[Dict]) -> None:
    fieldnames = [
        "event_slug",
        "event_title",
        "polymarket_date",
        "local_date",
        "local_time",
        "status",
        "play",
        "outcome",
        "label",
        "probability",
        "fair_odds",
        "risk",
        "condition",
        "top_score",
        "top_score_probability",
        "dominant_causal_layer",
        "dominant_influence_share",
        "usable_quote_count",
        "avg_spread",
        "total_liquidity",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for snapshot in snapshots:
            match = snapshot.get("match") or {}
            market_state = snapshot.get("market_state") or {}
            top_scores = ((snapshot.get("probabilities") or {}).get("top_scores") or [])
            top_score = top_scores[0] if top_scores else {}
            collapse_layers = snapshot.get("collapse_layers") or []
            dominant = collapse_layers[0] if collapse_layers else {}
            for path_item in snapshot.get("candidate_paths") or []:
                writer.writerow(
                    {
                        "event_slug": match.get("event_slug"),
                        "event_title": match.get("event_title"),
                        "polymarket_date": match.get("polymarket_date"),
                        "local_date": match.get("local_date"),
                        "local_time": match.get("local_time"),
                        "status": match.get("status"),
                        "play": path_item.get("play"),
                        "outcome": path_item.get("outcome"),
                        "label": path_item.get("label"),
                        "probability": path_item.get("probability"),
                        "fair_odds": path_item.get("fair_odds"),
                        "risk": path_item.get("risk"),
                        "condition": path_item.get("condition"),
                        "top_score": top_score.get("score"),
                        "top_score_probability": top_score.get("probability"),
                        "dominant_causal_layer": dominant.get("layer"),
                        "dominant_influence_share": dominant.get("influence_share"),
                        "usable_quote_count": market_state.get("usable_quote_count"),
                        "avg_spread": market_state.get("avg_spread"),
                        "total_liquidity": market_state.get("total_liquidity"),
                    }
                )


def print_schedule(rows: List[Dict], timezone_name: str) -> None:
    print(f"Polymarket schedule: matches={len(rows)} timezone={timezone_name}")
    for row in rows:
        print(
            f"{row.get('polymarket_date')} | {row.get('local_date')} {row.get('local_time')} "
            f"| {row.get('event_title')} | {row.get('status')} | {row.get('event_slug')}"
        )


def prune_live_files(cache_dir: Path, reports_dir: Path, active_slugs: set) -> None:
    for path in cache_dir.glob("fifwc-*.*"):
        slug = path.stem.replace("_probability", "", 1)
        if slug not in active_slugs:
            path.unlink(missing_ok=True)
    for path in reports_dir.glob("poly_match_fifwc-*.md"):
        slug = path.stem.replace("poly_match_", "", 1)
        if slug not in active_slugs:
            path.unlink(missing_ok=True)


def render_poly_match_report(matrix: EventMarketMatrix) -> str:
    match = MatchSP(
        match_id=matrix.match_id,
        date="live",
        home=matrix.home,
        away=matrix.away,
        spf_home=0.0,
        spf_draw=0.0,
        spf_away=0.0,
        handicap=0,
        rq_home=0.0,
        rq_draw=0.0,
        rq_away=0.0,
    )
    context = build_probability_context(match, matrix)
    lines = [
        f"# Polymarket 概率骨架：{matrix.home} vs {matrix.away}",
        "",
        f"- event: `{matrix.event_slug}`",
        f"- quote 数：{len(matrix.markets)}",
        "- 方法：胜平负/让球为核心约束，大小球/球队进球/BTTS/正确比分塑造比分分布；球员、首发、角球、远期按因果权重降权。",
    ]
    if has_extreme_quotes(matrix.markets):
        lines.append("- 警告：检测到 0/1 端点报价，可能是已结束、锁盘或流动性异常；该场只做概率记录，不应直接按 100% 作为投注依据。")
    if not has_usable_three_way_moneyline(matrix.markets):
        lines.append("- 警告：未检测到完整可交易胜平负三项盘口；主分支若出现概率，仅代表模型先验/残余盘口推导，不用于出票。")
    lines.extend(["", "## 因果层摘要", "", "| 因果层 | quote数 | 平均权重 |", "|---|---:|---:|"])
    for layer, item in sorted(causal_layer_summary(matrix.markets).items(), key=lambda kv: kv[1]["quotes"], reverse=True):
        lines.append(f"| {layer} | {int(item['quotes'])} | {item['avg_weight']:.2f} |")
    lines.extend(["", "## 归一化主分支", "", "| 分支 | 概率 | 公允赔率 |", "|---|---:|---:|"])
    for outcome, label in (("home", matrix.home), ("draw", "平局"), ("away", matrix.away)):
        prob = probability_for_spf(context, outcome) or 0.0
        lines.append(f"| {label} | {prob:.1%} | {fair_odds_text(prob)} |")
    lines.extend(["", "## 让球分支树（主队视角）", "", "| 让球 | 体彩分支 | 实际比分条件 | 概率 |", "|---:|---|---|---:|"])
    for handicap in (-3, -2, -1, 1, 2, 3):
        for outcome, label in (("home", "让胜"), ("draw", "让平"), ("away", "让负")):
            prob = probability_for_handicap(context, handicap, outcome, matrix.home, matrix.away) or 0.0
            condition = handicap_condition(matrix.home, matrix.away, handicap, outcome)
            lines.append(f"| {handicap:+d} | {label} | {condition} | {prob:.1%} |")
    lines.extend(["", "## Top 正确比分（模型归一化后）", "", "| 比分 | 概率 |", "|---|---:|"])
    for score, prob in sorted(context.score_distribution.probs.items(), key=lambda kv: kv[1], reverse=True)[:12]:
        lines.append(f"| {score[0]}-{score[1]} | {prob:.1%} |")
    lines.extend(["", "## 高概率 Poly 盘口（不等于有体彩价值）", "", "| 类别 | Outcome | 概率 | 权重 | spread |", "|---|---|---:|---:|---:|"])
    for quote in high_probability_quotes(matrix.markets):
        lines.append(
            f"| {quote.category} | {quote.outcome} | {quote.probability or 0.0:.1%} | "
            f"{quote.model_weight or 0.0:.2f} | {format_optional(quote.spread)} |"
        )
    return "\n".join(lines) + "\n"


def high_probability_quotes(quotes: Iterable[MarketQuote]) -> List[MarketQuote]:
    core_categories = {
        "moneyline",
        "handicap",
        "total_goals",
        "team_total",
        "btts",
        "starting_lineup",
        "halftime_result",
        "second_half_result",
    }
    candidates = [
        quote
        for quote in quotes
        if quote.category in core_categories
        and quote.probability is not None
        and quote.probability >= 0.60
        and quote.probability < 0.995
        and not quote.is_complement
        and quote.closed is not True
        and quote.accepting_orders is not False
    ]
    candidates.sort(key=lambda quote: ((quote.model_weight or 0.0), quote.probability or 0.0), reverse=True)
    return candidates[:18]


def fair_odds_text(probability: float) -> str:
    if probability <= 0:
        return "-"
    return f"{1.0 / probability:.2f}"


def has_extreme_quotes(quotes: Iterable[MarketQuote]) -> bool:
    core_categories = {"moneyline", "handicap", "total_goals", "team_total"}
    return any(
        quote.category in core_categories
        and quote.probability is not None
        and (quote.probability <= 0.005 or quote.probability >= 0.995)
        for quote in quotes
    )


def has_usable_three_way_moneyline(quotes: Iterable[MarketQuote]) -> bool:
    outcomes = {
        quote.outcome
        for quote in quotes
        if quote.category == "moneyline"
        and quote.outcome in {"home", "draw", "away"}
        and quote.probability is not None
        and quote.closed is not True
        and quote.accepting_orders is not False
    }
    return outcomes == {"home", "draw", "away"}


def format_optional(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{value:.3f}"


def default_world_cup_queries() -> List[str]:
    return [
        "World Cup",
        "World Cup More Markets",
        "FIFA World Cup More Markets",
        "FIFA World Cup spreads",
        "FIFA World Cup totals",
        "FIFA World Cup over under",
        "FIFA World Cup both teams to score",
        "FIFA World Cup correct score",
        "World Cup Winner",
        "World Cup Team to advance to Knockout Stages",
        "World Cup Group Winner",
        "World Cup Nation To Reach Quarterfinals",
        "World Cup Nation to Reach Final",
        "Germany Curaçao",
        "Netherlands Japan",
        "Ivory Coast Ecuador",
        "Sweden Tunisia",
    ]


def write_inventory_csv(path: str, rows: List[Dict]) -> None:
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "event_slug",
        "event_title",
        "polymarket_date",
        "start_time_utc",
        "event_active",
        "event_closed",
        "event_ended",
        "event_updated_at",
        "home",
        "away",
        "category",
        "sports_type",
        "scope",
        "period",
        "entity",
        "side",
        "line",
        "horizon",
        "causal_layer",
        "model_weight",
        "is_complement",
        "active",
        "closed",
        "accepting_orders",
        "question",
        "outcome",
        "probability",
        "fair_odds",
        "bid",
        "ask",
        "spread",
        "liquidity",
        "volume",
        "market_id",
        "token_id",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run(args: argparse.Namespace) -> int:
    matches = load_ticai_matches(args.sp_file)
    if args.html_file:
        matches.extend(load_ticai_matches(args.html_file))
    matches = [match for match in matches if match.date == args.date]
    if not matches:
        print(f"No matches found for {args.date}", file=sys.stderr)
        return 2

    cache_dir = Path("data/cache")
    polymarket_by_match: Dict[str, EventMarketMatrix] = {}
    if args.polymarket_cache:
        polymarket_by_match = load_matrices_from_file(args.polymarket_cache, matches)

    polymarket = PolymarketClient(
        cache_dir=cache_dir,
        offline=args.offline_cache,
        refresh=args.refresh_polymarket,
        cache_ttl_seconds=args.polymarket_cache_ttl,
    )
    facts_client = APIFootballClient(cache_dir=cache_dir, offline=args.offline_cache)

    analyses = []
    for match in matches:
        matrix = polymarket_by_match.get(match.match_id)
        if matrix is None:
            matrix = polymarket.discover_event(match, competition=args.competition)
        facts = facts_client.facts_for_match(match)
        analyses.append(analyze_match(match, matrix, facts))

    selections = flatten_selections(analyses)
    combo_groups = generate_combos(selections)
    allocated = allocate_stakes(combo_groups, args.budget)
    report = render_markdown_report(args.date, args.budget, analyses, allocated, combo_groups)
    report_out = args.report_out or f"reports/jc_research_{args.date}.md"
    path = write_report(report_out, report)
    print(f"Report written: {path}")
    return 0


# ---------------------------------------------------------------------------
# Harness subcommands (Phase 5)
# ---------------------------------------------------------------------------

def _load_outcomes(results_path: Optional[str]):
    """Load outcomes from a CSV or JSON file; detect by extension."""
    from ball_quant.adapters.results import load_results, load_results_json
    if results_path is None:
        return {}
    p = Path(results_path)
    if p.suffix.lower() == ".json":
        return load_results_json(str(p))
    return load_results(str(p))


def _load_snapshot_records(store_root: Path, date_from: str, date_to: str, match_id: Optional[str]) -> List[dict]:
    """List manifest entries and load their full snapshot records."""
    from ball_quant.data.store import list_snapshots, read_snapshot

    # list_snapshots compares ISO timestamps lexicographically; prefix with date to
    # cover any captured_at time within that calendar day.
    entries = list_snapshots(
        store_root,
        match_id=match_id,
        since=date_from,         # "2026-06-01" matches anything >= that string
        until=date_to + "T99",   # "T99" ensures the full end day is included
    )
    records = []
    for entry in entries:
        snap_path = Path(entry["path"])
        if snap_path.exists():
            records.append(read_snapshot(snap_path))
    return records


def cmd_capture(args: argparse.Namespace, settings: Settings) -> int:
    """Fetch an EventMarketMatrix for a slug and persist a bq.snapshot.v1 snapshot."""
    from ball_quant.data.capture import capture_snapshot

    store_root = Path(args.store_root or settings.store_root)

    # --- Build EventMarketMatrix ---
    if args.polymarket_cache:
        # Offline path: load from a pre-fetched JSON fixture.
        matrices = load_matrices_from_file(args.polymarket_cache, [])
        # Try to find the slug in the fixture.
        matrix = matrices.get(args.slug)
        if matrix is None:
            # Fall back: try match_id match on slug.
            for m in matrices.values():
                if m.event_slug == args.slug:
                    matrix = m
                    break
        if matrix is None:
            # Build a minimal EventMarketMatrix from slug alone (no quotes).
            home, away = infer_match_teams(args.slug)
            matrix = EventMarketMatrix(
                match_id=args.slug,
                home=home,
                away=away,
                event_id="",
                event_slug=args.slug,
                markets=[],
                raw_event={},
            )
    else:
        client = PolymarketClient(
            cache_dir=Path(settings.cache_dir),
            refresh=True,
            enrich_orderbook=args.with_orderbook,
            enrich_sports_payload=True,
        )
        event = client.prefer_sports_event(client.get_event_by_slug(args.slug), competition=args.competition)
        title = str(event.get("title") or args.slug)
        home, away = infer_match_teams(title)
        matrix = EventMarketMatrix(
            match_id=str(event.get("id") or event.get("slug") or args.slug),
            home=home,
            away=away,
            event_id=str(event.get("id") or ""),
            event_slug=event.get("slug") or args.slug,
            markets=event_to_quotes(event, home, away),
            raw_event=event,
        )
        if args.with_orderbook:
            client.enrich_with_clob(matrix, max_quotes=args.orderbook_limit)

    # --- Optionally load MatchSP ---
    match_sp: Optional[MatchSP] = None
    if args.sp_file:
        sp_matches = load_ticai_matches(args.sp_file)
        # Match by slug/match_id or fall back to first match in file.
        for m in sp_matches:
            if m.match_id == matrix.match_id or m.match_id == args.slug:
                match_sp = m
                break
        if match_sp is None and sp_matches:
            match_sp = sp_matches[0]

    snap_path = capture_snapshot(
        matrix=matrix,
        match_sp=match_sp,
        root=store_root,
        competition=args.competition,
    )
    print(f"Snapshot written: {snap_path}")
    return 0


def cmd_settle(args: argparse.Namespace, settings: Settings) -> int:
    """Load results CSV and persist to <store>/outcomes/results.json."""
    from ball_quant.adapters.results import load_results, save_results

    store_root = Path(args.store_root or settings.store_root)
    outcomes = load_results(args.results)

    out_path = Path(args.out) if args.out else store_root / "outcomes" / "results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_results(outcomes, str(out_path))

    settled = sum(1 for o in outcomes.values() if not o.void)
    voided = sum(1 for o in outcomes.values() if o.void)
    print(f"Settled: {settled}  Voided: {voided}  Total: {len(outcomes)}")
    print(f"Outcomes written: {out_path}")
    return 0


def cmd_backtest(args: argparse.Namespace, settings: Settings) -> int:
    """Replay snapshots in a date range against known outcomes and write a report."""
    from ball_quant.backtest.engine import run_backtest
    from ball_quant.backtest.report import render_backtest_report
    from ball_quant.core.params import DEFAULT_PARAMS, StrategyParams
    from ball_quant.core.profiles import ParamProfiles

    store_root = Path(args.store_root or settings.store_root)
    budget = args.budget if args.budget is not None else settings.default_budget
    bankroll = args.bankroll if args.bankroll is not None else settings.default_bankroll

    # Load StrategyParams from --params (JSON file or inline JSON string).
    params = DEFAULT_PARAMS
    if args.params:
        p = Path(args.params)
        raw_json = p.read_text(encoding="utf-8") if p.exists() else args.params
        params_dict = json.loads(raw_json)
        params = StrategyParams.from_dict(params_dict)

    # Load ParamProfiles from --profiles (mutually independent of --params;
    # when both are supplied, profiles takes precedence because it resolves
    # per-record — the flat params value is used only when profiles is None).
    profiles: Optional[ParamProfiles] = None
    if getattr(args, "profiles", None):
        profiles = ParamProfiles.from_json(args.profiles)

    records = _load_snapshot_records(store_root, args.date_from, args.date_to, args.match)
    outcomes = _load_outcomes(args.results)

    result = run_backtest(records, outcomes, params=params, budget=budget, bankroll=bankroll, profiles=profiles)
    title = f"Backtest {args.date_from} to {args.date_to}"
    report = render_backtest_report(result, title=title)

    report_out = args.report_out or f"reports/backtest_{args.date_from}_{args.date_to}.md"
    Path(report_out).parent.mkdir(parents=True, exist_ok=True)
    Path(report_out).write_text(report, encoding="utf-8")

    # Print a concise summary line for CI / ops.
    metrics = result.get("metrics", {})
    calib = metrics.get("calibration", {})
    pnl = metrics.get("pnl", {})
    brier = calib.get("brier")
    net_pnl = pnl.get("net_pnl")
    roi = pnl.get("roi")
    brier_str = f"{brier:.4f}" if brier is not None else "N/A"
    net_pnl_str = f"{net_pnl:.2f}" if net_pnl is not None else "N/A"
    roi_str = f"{roi:.2%}" if roi is not None else "N/A"
    print(
        f"records={result['n_records']} graded={result['n_graded_matches']} "
        f"brier={brier_str} net_pnl={net_pnl_str} roi={roi_str}"
    )
    print(f"Report written: {report_out}")
    return 0


def cmd_optimize(args: argparse.Namespace, settings: Settings) -> int:
    """Walk-forward parameter search over a snapshot date range."""
    from ball_quant.backtest.optimize import optimize_params, optimize_by_competition
    from ball_quant.backtest.report import render_optimization_report
    from ball_quant.core.profiles import ParamProfiles

    store_root = Path(args.store_root or settings.store_root)
    budget = args.budget if args.budget is not None else settings.default_budget
    bankroll = args.bankroll if args.bankroll is not None else settings.default_bankroll

    # Parse --space as JSON.
    space_path = Path(args.space)
    raw_space = space_path.read_text(encoding="utf-8") if space_path.exists() else args.space
    param_space = json.loads(raw_space)

    records = _load_snapshot_records(store_root, args.date_from, args.date_to, None)
    outcomes = _load_outcomes(args.results)

    # Walk-forward needs n_folds+1 time-ordered records. Surface an actionable message
    # instead of a raw ValueError from deep inside the split helper.
    if not records:
        print(
            f"No snapshots found in {args.date_from}..{args.date_to} under {store_root}; "
            "run `ballq capture` first.",
            file=sys.stderr,
        )
        return 2
    if len(records) < args.n_folds + 1:
        print(
            f"optimize needs >= n_folds+1 = {args.n_folds + 1} snapshots for "
            f"{args.n_folds}-fold walk-forward, but only {len(records)} are in range "
            f"{args.date_from}..{args.date_to}. Widen the range or pass --folds "
            f"{max(1, len(records) - 1)}.",
            file=sys.stderr,
        )
        return 2

    # --by-competition path: optimize per-competition, write ParamProfiles JSON.
    by_competition_flag = getattr(args, "by_competition", False)
    if by_competition_flag:
        result = optimize_by_competition(
            records=records,
            outcomes=outcomes,
            param_space=param_space,
            metric=args.metric,
            search=args.search,
            n_folds=args.n_folds,
            budget=budget,
            bankroll=bankroll,
            max_trials=args.max_trials,
            seed=args.seed,
        )

        profiles = ParamProfiles(
            default_overrides=result["default"],
            by_competition=result["by_competition"],
        )

        profiles_out = getattr(args, "profiles_out", None) or (
            (args.report_out.rsplit(".", 1)[0] if args.report_out else
             f"reports/profiles_{args.date_from}_{args.date_to}_{args.metric}") + "_profiles.json"
        )
        Path(profiles_out).parent.mkdir(parents=True, exist_ok=True)
        profiles.to_json(profiles_out)

        # Print summary: per-competition best + skipped.
        print(f"default_overrides={json.dumps(result['default'])}")
        for comp, overrides in result["by_competition"].items():
            detail = result["per_competition_detail"][comp]
            print(
                f"  competition={comp!r} "
                f"best_overrides={json.dumps(overrides)} "
                f"oos_{args.metric}={detail.get('best_out_of_sample')}"
            )
        for comp, reason in result["skipped"].items():
            print(f"  SKIPPED competition={comp!r}: {reason}")
        print(f"ParamProfiles written: {profiles_out}")
        return 0

    # Standard single-space optimize path (unchanged).
    opt = optimize_params(
        records=records,
        outcomes=outcomes,
        param_space=param_space,
        metric=args.metric,
        search=args.search,
        n_folds=args.n_folds,
        budget=budget,
        bankroll=bankroll,
        max_trials=args.max_trials,
        seed=args.seed,
    )

    title = f"Optimization {args.date_from} to {args.date_to} metric={args.metric}"
    report = render_optimization_report(opt, title=title)

    report_out = args.report_out or f"reports/optimize_{args.date_from}_{args.date_to}_{args.metric}.md"
    Path(report_out).parent.mkdir(parents=True, exist_ok=True)
    Path(report_out).write_text(report, encoding="utf-8")

    # Print best overrides + OOS score for CI.
    best_overrides = opt.get("best_overrides", {})
    best_oos = opt.get("best_out_of_sample")
    print(f"best_overrides={json.dumps(best_overrides)} oos_{args.metric}={best_oos}")
    print(f"Report written: {report_out}")
    return 0


# ---------------------------------------------------------------------------
# recommend — turnkey 体彩 betting slip
# ---------------------------------------------------------------------------

def _load_recommend_polymarket_matrices(polymarket_cache: Optional[str]) -> List[Any]:
    """Load Polymarket matrices from cache file into a flat list of EventMarketMatrix.

    Accepts two file formats:
      1. {matrices: [{home, away, markets, ...}, ...]}   — new recommend format
      2. {matches: {<match_id>: {home, away, markets, ...}}}  — legacy run format

    Returns [] when polymarket_cache is None (live path; caller handles it).
    """
    from ball_quant.models import EventMarketMatrix, MarketQuote

    if not polymarket_cache:
        return []

    payload = json.loads(Path(polymarket_cache).read_text(encoding="utf-8"))

    # Format 1: flat list under "matrices" key
    if "matrices" in payload:
        matrices = []
        for item in payload["matrices"]:
            quotes_raw = item.get("markets", [])
            # MarketQuote fields may have None for optional fields — pass only known keys.
            quotes = []
            for q in quotes_raw:
                q_clean = {k: v for k, v in q.items() if k in {
                    "market_id", "question", "category", "outcome", "probability",
                    "token_id", "bid", "ask", "spread", "liquidity", "volume",
                    "sports_type", "line", "period", "side", "entity", "scope",
                    "horizon", "causal_layer", "model_weight", "is_complement",
                    "active", "closed", "accepting_orders",
                }}
                quotes.append(MarketQuote(**q_clean))
            matrices.append(
                EventMarketMatrix(
                    match_id=item.get("match_id", ""),
                    home=item.get("home", ""),
                    away=item.get("away", ""),
                    event_id=item.get("event_id"),
                    event_slug=item.get("event_slug"),
                    markets=quotes,
                    raw_event=item.get("raw_event", {}),
                )
            )
        return matrices

    # Format 2: legacy {matches: {<id>: {...}}}
    by_match = payload.get("matches", {})
    matrices = []
    for match_id, item in by_match.items():
        quotes_raw = item.get("markets", [])
        quotes = []
        for q in quotes_raw:
            q_clean = {k: v for k, v in q.items() if k in {
                "market_id", "question", "category", "outcome", "probability",
                "token_id", "bid", "ask", "spread", "liquidity", "volume",
                "sports_type", "line", "period", "side", "entity", "scope",
                "horizon", "causal_layer", "model_weight", "is_complement",
                "active", "closed", "accepting_orders",
            }}
            quotes.append(MarketQuote(**q_clean))
        matrices.append(
            EventMarketMatrix(
                match_id=match_id,
                home=item.get("home", ""),
                away=item.get("away", ""),
                event_id=item.get("event_id"),
                event_slug=item.get("event_slug"),
                markets=quotes,
                raw_event=item.get("raw_event", {}),
            )
        )
    return matrices


def _apply_max_legs_filter(combo_groups: Dict, max_legs: int) -> Dict:
    """Post-filter: drop combos that exceed max_legs or mix hafu/correct_score beyond 3 legs.

    SIMPLIFIED subset of 竞彩 串关 rules:
      - Any combo with more than max_legs legs is rejected.
      - Combos containing a correct_score or hafu leg AND having more than 3 legs
        are rejected (体彩 per-play 单关 eligibility tightens for exact-margin plays).
    Finer per-玩法 single-leg eligibility (e.g. per-draw 单关 gates) is data-dependent
    and out of scope — this is the basic structural guard only.
    """
    from ball_quant.models import Combo

    _EXACT_PLAYS = {"correct_score", "hafu"}

    def _should_drop(combo: Combo) -> Optional[str]:
        n = len(combo.selections)
        if n > max_legs:
            return f"combo has {n} legs > max_legs={max_legs}"
        has_exact = any(s.play in _EXACT_PLAYS for s in combo.selections)
        if has_exact and n > 3:
            return f"combo mixes correct_score/hafu with {n} legs > 3-leg 串关 limit"
        return None

    # Accumulate dropped combos separately, then merge with existing "deleted"
    # list at the end.  Do NOT write to filtered["deleted"] during iteration
    # because the input's "deleted" key may appear in any order and would
    # overwrite combos we just dropped.
    extra_deleted: List[Any] = []
    filtered: Dict = {}
    for key, combos in combo_groups.items():
        if key == "deleted":
            # Collect the pre-existing deleted combos; merge at end
            filtered["deleted"] = list(combos)
            continue
        kept = []
        for combo in combos:
            reason = _should_drop(combo)
            if reason:
                combo.deletion_reason = (combo.deletion_reason or "") + f"; max-legs: {reason}"
                extra_deleted.append(combo)
            else:
                kept.append(combo)
        filtered[key] = kept
    # Merge pre-existing deleted list with newly dropped combos
    filtered.setdefault("deleted", [])
    filtered["deleted"].extend(extra_deleted)
    return filtered


def _render_recommend_markdown(
    date_str: str,
    budget: float,
    pairs_count: int,
    unmatched: List[Any],
    all_selections: List[Any],
    staked_combos: List[Any],
    total_staked: float,
    gated_out_count: int,
) -> str:
    """Render the betting slip as Markdown."""
    lines = [
        f"# 体彩竞彩推荐单 {date_str}",
        "",
        f"- **预算**: ¥{budget:.0f}",
        f"- **配对场次**: {pairs_count}  |  **未配对场次**: {len(unmatched)}",
        "- **方法**: 概率来自Polymarket实时盘口，赔率为体彩，"
        "edge=P×体彩赔率-1，90分钟基准",
        "",
        "## 推荐投注单 (单关)",
        "",
        "| 比赛 | 玩法 | 选择 | 体彩赔率 | P(模型) | edge | 投注额 | 预期回报 |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]

    # Singles from staked_combos (len==1)
    singles = [c for c in staked_combos if len(c.selections) == 1 and c.stake > 0]
    for combo in singles:
        sel = combo.selections[0]
        match_label = f"{sel.home} vs {sel.away}"
        expected_return = combo.stake * sel.probability * sel.sp
        lines.append(
            f"| {match_label} | {sel.play} | {sel.outcome} "
            f"| {sel.sp:.2f} | {sel.probability:.1%} | {sel.edge:+.2%} "
            f"| ¥{combo.stake:.0f} | ¥{expected_return:.0f} |"
        )

    if not singles:
        lines.append("| — | — | — | — | — | — | — | — |")

    # Parlays section
    parlays = [c for c in staked_combos if len(c.selections) > 1 and c.stake > 0]
    lines.extend(["", "## 串关推荐", ""])
    if parlays:
        lines.extend(["| 组合 | 腿数 | 串关赔率 | P(组合) | 投注额 | 预期回报 |", "|---|---:|---:|---:|---:|---:|"])
        for combo in parlays:
            legs = " × ".join(f"{s.play}:{s.outcome}" for s in combo.selections)
            expected_return = combo.stake * combo.probability * combo.odds
            lines.append(
                f"| {legs} | {len(combo.selections)} | {combo.odds:.2f} "
                f"| {combo.probability:.1%} | ¥{combo.stake:.0f} | ¥{expected_return:.0f} |"
            )
    else:
        lines.append("*(无串关推荐)*")

    expected_total = sum(c.stake * c.probability * c.odds for c in staked_combos if c.stake > 0)
    leftover = budget - total_staked
    lines.extend([
        "",
        "## 汇总",
        "",
        f"- **总投注额**: ¥{total_staked:.0f}",
        f"- **预期回报**: ¥{expected_total:.0f}",
        f"- **预算剩余**: ¥{leftover:.0f}",
        f"- **门控淘汰数**: {gated_out_count} 个选择 (edge≤0或流动性不足)",
        "",
        "## 未配对体彩场次 (无Polymarket对应 → 无法定价)",
        "",
    ])
    if unmatched:
        for u in unmatched:
            lines.append(f"- `{u.match_id}` {u.home} vs {u.away} ({u.match_date})")
    else:
        lines.append("*(全部场次已配对)*")
    return "\n".join(lines) + "\n"


def _build_recommend_json(
    date_str: str,
    budget: float,
    staked_combos: List[Any],
    unmatched: List[Any],
    total_staked: float,
) -> Dict:
    """Build structured JSON slip for LLM consumption."""
    bets = []
    for combo in staked_combos:
        if combo.stake <= 0:
            continue
        if len(combo.selections) == 1:
            sel = combo.selections[0]
            bets.append({
                "type": "single",
                "match": f"{sel.home} vs {sel.away}",
                "play": sel.play,
                "outcome": sel.outcome,
                "ticai_odds": round(sel.sp, 3),
                "prob": round(sel.probability, 4),
                "edge": round(sel.edge, 4),
                "stake": combo.stake,
                "expected_return": round(combo.stake * sel.probability * sel.sp, 2),
            })
        else:
            bets.append({
                "type": "parlay",
                "legs": [
                    {
                        "match": f"{s.home} vs {s.away}",
                        "play": s.play,
                        "outcome": s.outcome,
                        "ticai_odds": round(s.sp, 3),
                        "prob": round(s.probability, 4),
                        "edge": round(s.edge, 4),
                    }
                    for s in combo.selections
                ],
                "parlay_odds": round(combo.odds, 3),
                "parlay_prob": round(combo.probability, 4),
                "stake": combo.stake,
                "expected_return": round(combo.stake * combo.probability * combo.odds, 2),
            })
    return {
        "date": date_str,
        "budget": budget,
        "total_staked": total_staked,
        "recommended_bets": bets,
        "unmatched_ticai": [
            {"match_id": u.match_id, "home": u.home, "away": u.away, "date": u.match_date}
            for u in unmatched
        ],
    }


def cmd_bundle(args: argparse.Namespace, settings: Settings) -> int:
    """Emit the per-match RAW data bundle for the LLM analyst. No edge, no slip."""
    import json
    import logging as _logging

    from ball_quant.core import bundle as bundle_mod

    bundles, unmatched = bundle_mod.run_bundle(
        date=args.date,
        c500_cache=getattr(args, "c500_cache", None),
        kg_path=getattr(args, "kg_path", None),
    )
    md = bundle_mod.render_bundle_markdown(bundles, args.date)
    out_base = args.out or f"reports/bundle_{args.date}"
    json_path = Path(out_base + ".json")
    md_path = Path(out_base + ".md")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(bundles, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    md_path.write_text(md, encoding="utf-8")
    print(f"Bundle: {len(bundles)} matches paired, {len(unmatched)} 体彩 unmatched.")
    print(f"JSON: {json_path}")
    print(f"MD:   {md_path}")

    # --- persist forecast ledger (non-fatal) ---
    try:
        from ball_quant.core.forecast_ledger import append_forecast, make_forecast_record
        ledger_path = getattr(args, "forecast_ledger", None) or "data/forecasts/ledger.jsonl"
        n_saved = 0
        for entry in bundles:
            rec = make_forecast_record(entry)
            append_forecast(rec, ledger_path)
            n_saved += 1
        print(f"Forecast ledger: {n_saved} records appended → {ledger_path}")
    except Exception as _exc:  # noqa: BLE001
        _logging.getLogger(__name__).warning(
            "Forecast ledger append failed (non-fatal): %s", _exc
        )

    return 0


def cmd_grade(args: argparse.Namespace, settings: Settings) -> int:
    """Grade forecast ledger against actual results → calibration report per forecaster×market."""
    import logging as _logging

    from ball_quant.adapters.results import load_results
    from ball_quant.core.forecast_ledger import (
        calibration_report,
        grade_forecasts,
        load_forecasts,
    )

    _log = _logging.getLogger(__name__)

    ledger_path = args.ledger or "data/forecasts/ledger.jsonl"
    date_filter = getattr(args, "date", None) or None

    records = load_forecasts(ledger_path, date=date_filter)
    if not records:
        print(f"No forecast records found in {ledger_path}" +
              (f" for date {date_filter}" if date_filter else "") + ".")
        return 0

    outcomes = load_results(args.results)

    grouped, n_excluded = grade_forecasts(records, outcomes)
    report = calibration_report(grouped, n_excluded_post_kickoff=n_excluded)

    # --- build table ---
    header = f"{'Forecaster':<12} {'Market':<12} {'Brier':>8} {'LogLoss':>9} {'ECE':>8} {'N':>5}"
    sep = "-" * len(header)
    lines = [
        f"# Calibration Report",
        f"Ledger: {ledger_path}  |  Results: {args.results}",
        (f"Date filter: {date_filter}" if date_filter else "Date filter: all"),
        "",
        sep,
        header,
        sep,
    ]
    for row in report["rows"]:
        ece_str = f"{row['ece']:.6f}" if row["ece"] is not None else "    N/A"
        lines.append(
            f"{row['forecaster']:<12} {row['market_family']:<12} "
            f"{row['brier']:>8.6f} {row['log_loss']:>9.6f} {ece_str:>8} {row['n']:>5}"
        )
    lines.append(sep)
    lines.append("")
    lines.append(f"Poly vs Elo 1X2: {report['poly_vs_elo_1x2']}")
    lines.append(f"Excluded (post-kickoff): {report['n_excluded_post_kickoff']} records")

    output = "\n".join(lines)
    print(output)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output + "\n", encoding="utf-8")
        print(f"\nReport written to {args.out}")

    return 0


def cmd_recommend(args: argparse.Namespace, settings: Settings) -> int:
    """Generate a 体彩 betting slip for --budget using Polymarket as probability oracle."""
    from ball_quant.adapters.sporttery import load_odds, fetch_odds_raw, parse_odds
    from ball_quant.core.match_join import pair_all
    from ball_quant.core.ticai_engine import analyze_ticai, rank_recommendations, recommend_portfolio
    from ball_quant.core.params import DEFAULT_PARAMS, StrategyParams

    # ---- Load StrategyParams from --params ----
    params = DEFAULT_PARAMS
    if args.params:
        p = Path(args.params)
        raw_json = p.read_text(encoding="utf-8") if p.exists() else args.params
        params = StrategyParams.from_dict(json.loads(raw_json))

    bankroll = args.bankroll if args.bankroll is not None else args.budget

    # ---- Step 1: Load 体彩 odds ----
    if args.sporttery_cache:
        # Cassette or pre-captured API payload
        ticai_list = load_odds(Path(args.sporttery_cache))
    elif args.sp_file:
        # Existing ticai CSV via adapters.ticai (spf/handicap only)
        from ball_quant.adapters.ticai import load_ticai_matches
        sp_matches = load_ticai_matches(args.sp_file)
        # Convert MatchSP → TicaiOdds (spf + handicap; CRS/TTG/HAFU unavailable from CSV)
        from ball_quant.models import TicaiOdds
        ticai_list = [
            TicaiOdds(
                match_id=m.match_id,
                match_date=m.date,
                league="",
                home=m.home,
                away=m.away,
                match_num=None,
                spf={"home": m.spf_home, "draw": m.spf_draw, "away": m.spf_away},
                handicap_line=float(m.handicap) if m.handicap else None,
                rqspf={"home": m.rq_home, "draw": m.rq_draw, "away": m.rq_away},
                correct_score={},
                total_goals={},
                hafu={},
            )
            for m in sp_matches
        ]
    elif getattr(args, "c500_cache", None):
        # Offline cassette replay from a directory of c500_*.html files
        from ball_quant.adapters.c500 import load_odds as c500_load_odds
        ticai_list = c500_load_odds(
            cache_dir=Path(args.c500_cache),
            date=args.date,
        )
    elif getattr(args, "c500_live", False):
        # Live fetch from trade.500.com (requires network access to 500.com)
        from ball_quant.adapters.c500 import fetch_odds as c500_fetch_odds
        if not args.date:
            import datetime
            _today = datetime.date.today().strftime("%Y-%m-%d")
        else:
            _today = args.date
        ticai_list = c500_fetch_odds(date=_today)
    else:
        # Live sporttery fetch (will 567 outside China)
        raw = fetch_odds_raw()
        ticai_list = parse_odds(raw)

    # Optional date filter
    date_str = args.date or (ticai_list[0].match_date if ticai_list else "unknown")
    if args.date:
        ticai_list = [t for t in ticai_list if t.match_date == args.date]
        if not ticai_list:
            print(f"No 体彩 matches found for {args.date}", file=sys.stderr)
            return 2

    if not ticai_list:
        print("No 体彩 matches loaded — check source arguments.", file=sys.stderr)
        return 2

    # ---- Step 2: Load Polymarket matrices ----
    matrices: List[Any] = []
    if args.polymarket_cache:
        matrices = _load_recommend_polymarket_matrices(args.polymarket_cache)
    else:
        # Live Polymarket via PolymarketClient — use prefer_sports_event to fetch the
        # FULL per-match market set (moneyline + handicap + totals + correct_score +
        # btts + team_total + first_half_* etc.) the same way poly-match does.
        # WHY: fetch_world_cup_events returns only the gamma keyset with ~3 moneyline
        # quotes per match.  prefer_sports_event fetches the Polymarket Sports page
        # which carries ALL market categories; these calibrate the scoring grid even
        # though only 5 竞彩 玩法 are bet.
        client = PolymarketClient(
            cache_dir=Path("data/cache"),
            refresh=True,
            enrich_sports_payload=True,  # enables prefer_sports_event for each slug
        )
        from ball_quant.adapters.polymarket import event_to_quotes, infer_match_teams
        from ball_quant.models import EventMarketMatrix
        events = client.fetch_world_cup_events(tag_id=WORLD_CUP_TAG_ID, max_events=700, include_closed=False)
        for event in events:
            slug = event.get("slug") or ""
            # Enrich each match-level event with the full Sports-page market set.
            # prefer_sports_event is a no-op for non-match slugs so it is safe to
            # call unconditionally; it logs a warning and returns the original event
            # on network failure.
            enriched_event = client.prefer_sports_event(event)
            title = str(enriched_event.get("title") or enriched_event.get("slug") or "")
            home, away = infer_match_teams(title)
            matrix = EventMarketMatrix(
                match_id=str(enriched_event.get("id") or enriched_event.get("slug") or ""),
                home=home,
                away=away,
                event_id=str(enriched_event.get("id") or ""),
                event_slug=enriched_event.get("slug") or "",
                markets=event_to_quotes(enriched_event, home, away),
                raw_event=enriched_event,
            )
            matrices.append(matrix)

    # ---- Step 3: Pair 体彩 ↔ Polymarket by team name ----
    matched_pairs, unmatched = pair_all(ticai_list, matrices, date_tolerance_days=1)

    if not matched_pairs:
        print(
            f"Warning: 0 matched pairs. {len(unmatched)} unmatched 体彩 matches.",
            file=sys.stderr,
        )
        # Still write outputs so tests can inspect the unmatched list
    else:
        print(f"Paired {len(matched_pairs)} matches; {len(unmatched)} unmatched.")

    # ---- Steps 4–5: Per-match analyze + rank ----
    all_selections: List[Any] = []
    gated_out_count = 0

    for ticai, matrix in matched_pairs:
        selections, _skipped = analyze_ticai(ticai, matrix, params=params)
        rank_result = rank_recommendations(
            selections, matrix, params=params, min_edge=args.min_edge
        )
        ranked = rank_result["ranked"]
        gated_out_count += len(rank_result["gated_out"])
        all_selections.extend(ranked)

    # ---- Step 6: Portfolio allocation with leg-limit filter ----
    if not all_selections:
        print(
            "No selections passed edge gate — no bets to place. "
            f"Unmatched: {len(unmatched)}",
            file=sys.stderr,
        )
        staked_combos = []
        total_staked = 0.0
        combo_groups: Dict = {"A": [], "B": [], "C": [], "deleted": []}
    else:
        portfolio = recommend_portfolio(all_selections, budget=args.budget, params=params)
        staked_combos = portfolio["combos"]
        total_staked = portfolio["total_stake"]
        # Apply 串关 leg-limit post-filter (does NOT re-run generate_combos)
        # We regenerate combo_groups solely to apply the max-legs filter;
        # allocate_stakes is already done above and staked_combos reflects it.
        # For the purpose of leg-limit filtering: remove over-length combos from output.
        from ball_quant.core.combo import generate_combos as _gen_combos
        raw_groups = _gen_combos(all_selections, params=params)
        combo_groups = _apply_max_legs_filter(raw_groups, max_legs=args.max_legs)
        # Re-filter staked_combos to remove any that violate max-legs
        _EXACT_PLAYS = {"correct_score", "hafu"}
        kept_staked = []
        for combo in staked_combos:
            n = len(combo.selections)
            if n > args.max_legs:
                gated_out_count += 1
                continue
            has_exact = any(s.play in _EXACT_PLAYS for s in combo.selections)
            if has_exact and n > 3:
                gated_out_count += 1
                continue
            kept_staked.append(combo)
        staked_combos = kept_staked
        total_staked = sum(c.stake for c in staked_combos)

    # Clamp: total_staked must never exceed budget (staking.trim_to_budget should handle this,
    # but we guard again here because the leg-limit filter may have reduced the set further).
    if total_staked > args.budget + 0.01:
        print(f"Warning: total_staked {total_staked:.2f} > budget {args.budget:.2f}", file=sys.stderr)

    # ---- Output ----
    date_label = date_str.replace("-", "") if date_str != "unknown" else "unknown"
    report_path = Path(args.report_out or f"reports/recommend_{date_label}.md")
    json_path = Path(args.json_out or f"reports/recommend_{date_label}.json")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_text = _render_recommend_markdown(
        date_str=date_str,
        budget=args.budget,
        pairs_count=len(matched_pairs),
        unmatched=unmatched,
        all_selections=all_selections,
        staked_combos=staked_combos,
        total_staked=total_staked,
        gated_out_count=gated_out_count,
    )
    report_path.write_text(report_text, encoding="utf-8")

    json_payload = _build_recommend_json(date_str, args.budget, staked_combos, unmatched, total_staked)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(json_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # Concise stdout summary — readable in terminal, parseable by LLM operator
    print(f"=== 体彩推荐单 {date_str} | 预算 ¥{args.budget:.0f} ===")
    singles_out = [c for c in staked_combos if len(c.selections) == 1 and c.stake > 0]
    parlays_out = [c for c in staked_combos if len(c.selections) > 1 and c.stake > 0]
    if singles_out:
        print(f"单关 ({len(singles_out)}):")
        for c in singles_out:
            sel = c.selections[0]
            print(
                f"  {sel.home} vs {sel.away} | {sel.play}:{sel.outcome} "
                f"| 赔率{sel.sp:.2f} | P={sel.probability:.1%} | edge={sel.edge:+.2%} | ¥{c.stake:.0f}"
            )
    if parlays_out:
        print(f"串关 ({len(parlays_out)}):")
        for c in parlays_out:
            legs = " × ".join(f"{s.play}:{s.outcome}" for s in c.selections)
            print(f"  [{legs}] | 赔率{c.odds:.2f} | ¥{c.stake:.0f}")
    print(f"总投注额: ¥{total_staked:.0f} / 预算 ¥{args.budget:.0f} | 未配对: {len(unmatched)}")
    print(f"报告: {report_path}")
    print(f"JSON: {json_path}")
    return 0


def cmd_kg_build(args: argparse.Namespace) -> int:
    """Build / refresh the KG from Polymarket futures quotes.

    Offline path (--poly-dump): reads the JSON produced by `ballq poly-dump`,
    extracts tournament_winner / group_winner / group_advancement quotes from
    every event's market list, deviggs them, and upserts into the KG.

    Live path (--live): calls PolymarketClient.fetch_world_cup_events and does
    the same.  Intended for scheduled server runs.
    """
    from ball_quant.core.knowledge_graph import load_kg, save_kg
    from ball_quant.core.team_strength import update_kg_from_futures

    kg_path = Path(args.kg_out)
    kg = load_kg(kg_path)

    quotes: List[MarketQuote] = []

    if args.poly_dump:
        payload = json.loads(Path(args.poly_dump).read_text(encoding="utf-8"))
        # poly-dump format: {"events": [inventory, ...]}; each inventory has
        # "markets" list of quote-dicts.  Re-hydrate into MarketQuote objects.
        for inv in payload.get("events", []):
            for q in inv.get("markets", []):
                try:
                    quotes.append(MarketQuote(**{k: v for k, v in q.items() if k in MarketQuote.__dataclass_fields__}))
                except (TypeError, KeyError):
                    continue
    else:
        # Live fetch
        client = PolymarketClient(enrich_orderbook=False, enrich_sports_payload=False)
        events = client.fetch_world_cup_events(
            tag_id=args.world_cup_tag_id,
            max_events=args.max_world_cup_events,
        )
        for event in events:
            home, away = infer_match_teams(str(event.get("title") or event.get("slug") or ""))
            quotes.extend(event_to_quotes(event, home, away))

    # Filter to futures categories only — we don't want to push match-level
    # quotes (moneyline / handicap) into the team-strength derivation.
    futures_categories = {"tournament_winner", "group_winner", "group_advancement"}
    futures_quotes = [q for q in quotes if q.category in futures_categories]

    update_kg_from_futures(futures_quotes, kg, devig=args.devig)
    save_kg(kg, kg_path)

    n_teams = len(kg)
    print(f"kg-build: {n_teams} teams in KG, {len(futures_quotes)} futures quotes processed")
    print(f"KG written: {kg_path}")

    # Print top-10 by strength_win
    ranked = sorted(
        [(name, t) for name, t in kg.items() if t.strength_win is not None],
        key=lambda x: -(x[1].strength_win or 0.0),
    )[:10]
    if ranked:
        print("\nTop-10 by strength_win (devigged tournament-winner probability):")
        for i, (name, t) in enumerate(ranked, 1):
            adv = f"  advance={t.strength_advance:.3f}" if t.strength_advance is not None else ""
            print(f"  {i:2d}. {name:<20s}  win={t.strength_win:.4f}{adv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
