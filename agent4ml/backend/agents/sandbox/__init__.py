"""Sandbox 系统 — Agent 代码/文件操作的执行边界。

方案 C 组合模式：Sandbox 具体类组合 SandboxRuntime + PathTranslator + SecurityGuard
三可替换组件，负责编排（validate → translate → execute → mask）。

详见 design_docs/25-sandbox-overview.md + 26-sandbox-stage1-core-abstraction.md。

INVARIANT（Stage 1 范围）:
- merge_sandbox 幂等: 同 sandbox_id 合并 OK，不同 id fail-closed 抛 ValueError（bug 不静默）
- merge_sandbox 不清空: sandbox_id 是 thread 级持久状态，new=None 返 existing，无主动清空语义
- merge_sandbox 失败传播: reducer 抛 ValueError 保持，graph 外层捕获转优雅提示（Stage 4 实现）
- Sandbox 编排顺序: guard.validate → translator.translate → runtime.execute → translator.mask
- Sandbox 是具体类非 ABC: 有真实编排逻辑，组合三组件
- mask_output 归 PathTranslator: 脱敏是路径翻译逆操作（Grill #4），Guard 只做 validate
- SandboxRuntime 只裸执行: 不知路径翻译、不做安全检查（异构差异封装在 adapter 内）
- SandboxRuntime 异常统一: 全抛 SandboxError 子类，runtime 实现负责包装内置异常
- SecurityGuard 只做 validate: validate_path / validate_command，无 mask_output（Grill #4）
- SandboxInfo.created_at 用 ISO 格式: utc_now_iso，与 Agent4ML journal 统一
"""
