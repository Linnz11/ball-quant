# REFACTOR PLAN — 代码=数据服务，LLM=分析师
> 2026-06-17 · 触发：用户指出 `recommend` 输出是「代码算的结果」，不是模型分析。
> 北极星纠正 = 代码搬砖、模型思考。

## 0. 北极星（验收判据，不可违反）
- **代码**：采集原始数据 + 配对 + （可选）联合分布参考层。**绝不算 edge、绝不出票当答案。**
- **LLM（控制器 SKILL）= 唯一决策者**：读 bundle → 剧本 + 三层概率 + 薄盘怀疑 + A/B/C + 诚实 edge → 买法。
- **判据**：`recommend` 的代码裸 edge **不能**是产品答案；产品答案 = LLM 对 bundle 的分析。
- 反例（必须杜绝）：今晚那张 ¥10/波黑+61% — 代码裸数字、无 oracle 判断、薄盘幻觉。

## 1. 要改什么（具体到文件/行为）
1. **暴露体彩 单关/过关 flags**（数据缺口，每次靠手动核）
   - `adapters/c500.py` + `models.py(TicaiOdds)`：解析 `data-subactive`，每玩法加 `单关_open / 过关_open`。
2. **建 `bundle` 命令（核心 = 数据服务出口）**
   - 新 `core/bundle.py` + `cli.py` 子命令 `bundle --date`。
   - 每场（已修 match_join 配对）吐：
     - Poly 全角度去水概率 + **流动性/薄盘 flag**（oracle 质量信号）
     - 体彩 5 玩法赔率 + 单关/过关/未开售 flags
     - KG（实力/伤停）
     - （可选）Poisson 联合网格 = "代码参考层"（Codex 的 within-match joint model）
   - **不算 edge、不出票。** 输出 = LLM 可读 JSON + 紧凑 markdown。
3. **降级 `recommend`（代码出票那条）**
   - ticai_engine 的 edge/combo/staking **不再是产品答案** → 标"代码候选/粗筛"，移出决策路径。
   - **保留** probability grid 当 bundle 参考层。
4. **控制器接 bundle（LLM 当分析师）**
   - SKILL §2 工作流：从"3 并行采集 agent"改成"**跑一条 `bundle` → 我直接读 → 推理**"——杀掉每次手动起 agent 的慢。
5. **清理**
   - `.gitignore` 加 `data/cache/` `data/store/`；清空 6 个死 re-export `__init__` stub（不删文件，避免 import 断）。
6. **skill 重构（按 Agent C）**
   - 148→~90 行；去 7 处重复（R1–R30 各留一处）；**显式解"满买 vs 默认不下注"矛盾**（没 edge→娱乐总额；有 edge 才动 B 层；A 层始终兜底）；会话比分叙事移 `CHANGELOG.md`。
7. **闭环脊柱（later）**
   - `settle`/grade + ledger（PnL/CLV/校准）——验证到底有没有真 edge。

## 1b. 概率网格校准 spec（升级 `core/probability.py`，喂 bundle 的 grid 参考层）
> 研究定稿（我的框架 + Codex xhigh）。现有已是 bivariate-Poisson+IPF+Shin；本节是精度升级。
- **去水 + 可靠性权重**：mid=(bid+ask)/2；多 devig map（Shin/power/logit），跨 map 方差进 σ²；`σ²_g = c·spread² + c/log(1+depth) + c/log(1+vol) + c·age + c·devig方差 + σ_floor²`；`α_g = (1/σ²_g)·exp(−age/half_life)`，**单盘封顶 + 族内封顶**（Σα within 总进球/让球 ≤ family_cap——相邻线高度相关，别当独立）。
- **先验 q0**：独立 Poisson(λ_h,λ_a) × **Dixon-Coles ρ** 低分修正（修 0-0/1-0/0-1/1-1+平局质量）；λ 从流动 1X2+大小球初始化。bivariate-Poisson λ3 只在有历史校准集时开（默认关）。DC(低分依赖) 与 λ3(进球相关+总方差) **不是替代品**。
- **校准（软约束，非 IPF）**：`min KL(q‖q0) + Σ_g α_g·loss_g(B_g q, p_g)`；多元盘 loss=KL，二元盘 loss=**Huber-logit**（KL 对坏盘过度反应）；**指数梯度镜像下降**求解（`q ← q·exp(−η_t·grad)`+归一，stdlib 无 numpy，保正+归一）。
- **阶梯（精度核心）**：喂**全部让球档**→还原净胜分布 P(D=d)；**全部大小球档**→还原进球均值/方差/尾部。胜过单线。
- **尾部**：显式 8+ 桶，**不静默截断重归一**（比分层敏感）。薄盘向 q0 收缩：`p_used=β·p_poly+(1−β)·B_g q0`。
- **输出给 LLM**：各投影(比分/总进球/净胜/1X2/半全场) + **标准化残差 z_g=|B_g q−p_g|/σ_g** + market_influence + thin/stale flag + **q_band** + no_bet_reason。
- **铁律（最深）**：**自洽 ≠ 准确**。镜像下降永远吐自洽网格，但可能是噪声的优雅缝合。精度在"可靠性+依赖结构"加权（族封顶/时效/devig方差），**不在求解器**。流动族大残差 = 信号候选（体彩错价?），非真理；信"多个独立流动族一致"处。
- **验收**：**按玩法族**分别回测校准（Brier/log-loss/校准回归 per 1X2/大小球/让球/比分），非全局；（later）对收盘价评估。

## 2. 执行顺序 + 验收
| 步 | 做 | 验收当 |
|---|---|---|
| 1 | cleanup（gitignore + stub） | git tree 干净、import 不断、35+ 测试绿 |
| 2 | c500 暴露 flags | bundle 每玩法带 单/过/off，对得上手动核实值 |
| 3 | 建 `bundle` 命令 | `ballq bundle --date` 吐今晚 N 场干净原始数据（含薄盘 flag） |
| 4 | 控制器接 bundle | 我读 bundle → 出真·分析（剧本+A/B/C+诚实 edge），不是代码裸数字 |
| 5 | skill 重构 | ~90 行、矛盾解、R1–R30 全在唯一位置 |
| 6（later） | ledger | 每注进账本、可算校准/ROI |

## 3. 谁做 / 怎么做
- **我直接做**：cleanup、c500 flags、bundle、控制器接线、skill 重构（小 + 精确 + 可靠）。
- **Codex worker**（量大时）：ledger/settle。
- 在 `refactor/data-service` 分支；**每步一 commit**；最后设 remote 推送。

## 4. 已完成（前置）
- ✅ 审计（3 agent）：代码是干净 DAG、无垃圾脚本；端到端断点=match_join；skill 臃肿(R1–R30 清单)。
- ✅ baseline 快照提交。
- ✅ **解锁键**：match_join 别名 + slug 取比赛日 → 配对 2→12、管线第一次出真票（commit d06279a）。
