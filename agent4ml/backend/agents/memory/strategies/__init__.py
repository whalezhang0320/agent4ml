"""记忆策略族。

default/ 是默认实现（Markdown + Ebbinghaus + BM25），未来可加 custom/ / vector/
平级扩展（00 §7.6 + 48 §2.2）。

策略可插拔：换衰减模型只换 DecayPolicy 实现，换检索策略只换 Retriever 实现，
换持久化形态只换 MemoryStore 实现（00 §7.8 解耦判据）。
"""
