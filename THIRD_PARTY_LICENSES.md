# Third-Party Licenses

This project includes skills adapted from the following open-source projects.
Original copyright notices are preserved per the MIT license terms.

## hermes-agent

- **Source**: https://github.com/NousResearch/hermes-agent (or its successor)
- **License**: MIT
- **Copyright**: Copyright (c) 2025 Nous Research
- **Skills adapted**: plan, systematic-debugging, test-driven-development,
  requesting-code-review, simplify-code, spike, github-code-review,
  skill-authoring, arxiv, research-paper-writing, osint-investigation,
  blogwatcher, node-inspect-debugger, python-debugpy, github-auth,
  github-issues, github-pr-workflow, github-repo-management,
  codebase-inspection, subagent-driven-development, concept-diagrams,
  architecture-diagram

## deer-flow

- **Source**: https://github.com/bytedance/deer-flow
- **License**: MIT
- **Copyright**: Copyright (c) 2025 Bytedance Ltd. and/or its affiliates;
  Copyright (c) 2025-2026 DeerFlow Authors
- **Skills adapted**: skill-creator, find-skills, bootstrap,
  academic-paper-review, systematic-literature-review, deep-research,
  github-deep-research, consulting-analysis, data-analysis,
  newsletter-generation, code-documentation, ppt-generation,
  chart-visualization, frontend-design

## Adaptation notes

Adapted skills have been modified from their originals:

- Frontmatter normalized to Agent4ML's schema (`name` / `description` /
  `allowed-tools` / `enabled` / `related-skills` / `license` / `author`).
- Tool references remapped to Agent4ML's tool set
  (`terminal` → `bash`, `search_files` → `grep`, `patch` → `str_replace`,
  `web_extract`/`web_search` → `web_search`, `read_file` → `read_file`,
  `find`/`ls` → `list_dir`, `cat`/`head`/`tail` → `read_file`).
- Project-specific paths rewritten (`.hermes/` → `.agent4ml/`).
- Hermes-private frontmatter fields (`metadata.hermes.*`, `platforms`,
  `version`) dropped or converted to Agent4ML equivalents.

Original skill content is preserved under each skill directory's `SKILL.md`
with the `author` field crediting the source project. See the upstream
repositories for the unmodified originals.
