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
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "doctor":
        return doctor(args)
    if args.command == "poly-dump":
        return poly_dump(args)
    if args.command == "poly-match":
        return poly_match(args)
    if args.command == "poly-schedule":
        return poly_schedule(args)
    if args.command == "auto-refresh":
        return auto_refresh(args)
    if args.command == "run":
        return run(args)
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


if __name__ == "__main__":
    raise SystemExit(main())
