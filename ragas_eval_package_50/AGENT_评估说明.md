# Agent 评估说明

## 这次代码已补齐的能力

当前评估脚本和 API 已支持输出并保存以下 Agent 相关字段：

- `source_type`
  实际答案来源类型，例如：`知识库` / `网络`

- `route`
  Agent 实际决策路径，例如：`仅 RAG（网络兜底未启用）`、`RAG（置信度 ...）`、`网络兜底（...）`

- `retrieval_confidence`
  本地 RAG 的检索置信度

- `contexts`
  实际用于回答的上下文
  - 走知识库时：返回检索 chunk
  - 走网络时：返回拼接后的搜索结果片段

## 你需要自己补的字段

如果你要评估“Agent 路由是否正确”，需要在原始题库 `ragas_eval_50.jsonl` 中为样本补标注。

建议补这两个可选字段：

### 1. `expected_source_type`

表示这条问题“理想情况下应该由哪个来源回答”。

可选示例：

```json
"expected_source_type": "知识库"
```

或：

```json
"expected_source_type": "网络"
```

### 2. `expected_route`

如果你想更细粒度地判断具体策略，也可以加这个字段。

可选示例：

```json
"expected_route": "rag"
```

或：

```json
"expected_route": "web"
```

## 推荐最小标注方式

为了减少工作量，建议你优先只补：

```json
"expected_source_type": "知识库"
```

或：

```json
"expected_source_type": "网络"
```

这样就足够先评估 Agent 的“来源决策是否正确”。

## 评估结果会新增什么

当输出记录里存在：

- `expected_source_type`
  或
- `expected_route`

脚本在最终 summary 中会额外统计：

- `route_eval_count`
- `route_match_count`
- `route_accuracy`

## 建议流程

1. 先给题库补 `expected_source_type`
2. 用 `--mode api` 跑整套 Agent 输出采集
3. 再跑 RAGAS 评分
4. 同时看：
   - 回答质量指标
   - 路由命中率 `route_accuracy`
