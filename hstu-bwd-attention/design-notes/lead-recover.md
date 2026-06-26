# Lead 恢复任务(pane 0.0)

你是 HSTU bwd GPU 实现项目的 **lead(pane 0.0)**,刚重建 tmux、上下文为空。请按下列顺序恢复工作状态,**只读不改**,完成后给我一句话现状汇报 + 下一步建议,等我裁决,先不要派单。

## 1. 必读(按序)
1. `/root/workspace/hstu-bwd-impl/docs/HANDOFF.md` —— 权威恢复文档(环境/构建 BUILD_DEV=OFF/tmux 派单 Enter-吞坑/铁律/已完成 M0–M6b/下一步候选/教训)。
2. `/root/workspace/hstu-bwd-impl/candidates.jsonl` —— 候选账本(12 行,promoted/pass/fail)。
3. `/tmp/hstu-bwd-design/M6b-done.md` —— 最近里程碑报告(group determ + 修 O1 + 修 harness bug)。

## 2. 核对当下基线
- git head 应为 `d4fb2884`(M6b)。`git -C /root/workspace/ck_hstu log --oneline -1` 核对。
- **暂不重跑测试套件**(跑全套要先 build,耗时);若我后续要求再跑 `python3 /root/workspace/hstu-bwd-impl/test/run_bwd_tests.py`(预期 91/91/0/0 exit 0)。

## 3. 团队布局(已重建)
- pane 0.0 = 你(lead);0.1 = 主 coder;0.2 = 文档/review;0.3 = review 备用。
- 派单铁律(HANDOFF §2):Write prompt→`tmux send-keys -t claudeteam:0.N "..." Enter`→**回读 capture-pane 确认输入框清空**;pane 忙时 Enter 会被吞。

## 4. 汇报格式(完成后输出)
- 一句话:当前能力边界 + git 里程碑。
- 下一步候选(M7 / cross-attention / M8 perf)各自就绪度与并行可行性(M6/M7 弱耦合可 worktree 真并行)。
- **不要自行开工或派单**,等 lead 用户(我)裁决。
