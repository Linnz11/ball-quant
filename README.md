# ball-quant

`ball-quant` 是一套每日可运行的竞彩盘口研究系统。它不是单场玄学预测器，而是把 Polymarket 市场概率、体彩 SP、球队事实、玩法映射、EV/Kelly、组合概率和仓位优化串成一条可复现流水线。

## 功能

- Polymarket event/market 发现与全盘口矩阵读取
- World Cup tag/keyset 全量玩法盘点，默认使用 Polymarket Gamma tag `102232`
- Sports 页面完整盘口补抓，覆盖比赛盘口、上下半场、角球、进球、助攻、射门等玩法
- 体彩 CSV/HTML 输入解析
- API-Football 球队事实层
- 普通胜平负、让球胜平负的比分条件映射
- 公允赔率、break-even、edge、Kelly、流动性惩罚、置信度
- 因果层权重：核心赛果最高，比分形态其次，球员/非进球道具更低，远期玩法按时间跨度降权
- 分支树、组合枚举、低质量组合删除
- 高概率配平版、RR 优化版、高赔率小搏版仓位分配
- Markdown 报告与体彩店口播

## 快速开始

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e .
python -m unittest
ballq run --date 2026-06-14 --budget 200 --sp-file examples/sample_jc.csv --offline-cache
```

实时盘点 Polymarket 世界杯全量玩法：

```bash
ballq poly-dump \
  --json-out data/cache/polymarket_worldcup_inventory.json \
  --csv-out data/cache/polymarket_worldcup_inventory.csv
```

单场比赛如果要展开 Sports 页面里的让球、大小球、球员射门/进球/助攻、角球等完整矩阵：

```bash
ballq poly-dump \
  --slug fifwc-nld-jpn-2026-06-14 \
  --slug-only \
  --json-out data/cache/poly_nld_jpn_sports.json \
  --csv-out data/cache/poly_nld_jpn_sports.csv
```

按 Polymarket 日期或北京时间列核心比赛赛程：

```bash
ballq poly-schedule --date 2026-06-15 --date-mode poly --timezone Asia/Shanghai
ballq poly-schedule --date 2026-06-16 --date-mode local --timezone Asia/Shanghai
```

每小时自动化任务应调用单次刷新命令。它会重写 active schedule，只保留未过期比赛，并刷新未来 36 小时内的 full matrix 与 Markdown 报告：

```bash
ballq auto-refresh --timezone Asia/Shanghai --lookahead-hours 36
```

输出位置：

```text
data/cache/poly_worldcup_active_schedule.json
data/cache/poly_worldcup_active_schedule.csv
data/cache/live/
reports/live/
```

其中 `data/cache/live/*_probability.json` 是每场实时概率塌缩快照，包含：

- `signal_layers`：全盘口信号层，含球员、角球、首发等低权重信号
- `collapse_layers`：真正参与比分分布塌缩的胜平负、让球、大小球、球队进球、BTTS、正确比分约束
- `collapse_constraints`：每条约束的 target、prior、final、strength、gap_before、gap_after
- `probabilities`：胜平负、让球分支、总进球、球队进球、BTTS、Top 比分、公允赔率
- `candidate_paths`：可用于串关枚举的实时概率路径

汇总 CSV：

```text
data/cache/live/live_probability_summary.csv
```

输出报告默认写入：

```text
reports/jc_research_YYYY-MM-DD.md
```

## 体彩 CSV 字段

最小字段：

```csv
match_id,date,home,away,spf_home,spf_draw,spf_away,handicap,rq_home,rq_draw,rq_away
```

`handicap` 使用主队视角，例如荷兰让 1 球写 `-1`，科特迪瓦受让 1 球写 `1`。

## API key

API-Football 使用环境变量：

```bash
export API_FOOTBALL_KEY="..."
```

没有 key 时系统会降级，只输出数据缺口和低置信度事实摘要。

## 重要提示

本项目只做研究和投注组合建模，不承诺收益。报告中的“保留/删除/仓位”是基于输入数据、市场价格和模型约束生成的风控建议。
