"""默认记忆策略组合（Markdown + Ebbinghaus + BM25）。

Layer 1 仅骨架：strategy.py 空入口 + _constants.py 占位。
Layer 2 填充 decay / forget / manager。
Layer 3 填充 store / retriever。

目录结构与 `agents/context_engineering/strategies/default/` 同构（48 §2.2）：
- strategy.py     默认策略主入口（组合各组件）
- _constants.py   常量（衰减参数 / 阈值 / BM25 权重）
- decay.py        EbbinghausDecayPolicy            [Layer 2 实现]
- forget.py       CompositeForgetPolicy            [Layer 2 实现]
- manager.py      DefaultMemoryManager（四操作）   [Layer 2 实现]
- store.py        MarkdownFileStore                [Layer 3 实现]
- retriever.py    HybridRetriever（组合 optional V/G） [Layer 3 实现]
"""
