from agent4ml.backend.agents.leader.prompts import apply_prompt_template


def test_default_mode_prompt() -> None:
    prompt = apply_prompt_template(expert_mode=False)
    assert "Agent4ML" in prompt
    assert "<identity>" in prompt
    assert "<constraints>" in prompt
    assert "<decision_guidance>" in prompt
    # default 模式不含 expert 段
    assert "<mode name=\"expert\">" not in prompt


def test_expert_mode_prompt() -> None:
    prompt = apply_prompt_template(expert_mode=True)
    assert "Agent4ML" in prompt
    assert "<identity>" in prompt
    assert "<constraints>" in prompt
    assert "<decision_guidance>" in prompt
    # expert 模式追加 mode_expert 段
    assert "<mode name=\"expert\">" in prompt
    assert "reflection" in prompt


def test_expert_mode_appends_expert_section() -> None:
    """expert prompt 应比 default 多 mode_expert 段。"""
    default_prompt = apply_prompt_template(expert_mode=False)
    expert_prompt = apply_prompt_template(expert_mode=True)
    assert len(expert_prompt) > len(default_prompt)
    assert "<mode name=\"expert\">" in expert_prompt
    assert "<mode name=\"expert\">" not in default_prompt


def test_context_kwargs_accepted() -> None:
    prompt = apply_prompt_template(expert_mode=False, skills=["a"], deferred_names=["b"])
    assert "Agent4ML" in prompt


def test_language_constraint_in_prompt() -> None:
    prompt = apply_prompt_template(expert_mode=False)
    assert "语言约束" in prompt or "language" in prompt.lower()
