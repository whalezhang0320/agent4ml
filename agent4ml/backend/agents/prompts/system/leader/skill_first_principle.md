# Skill-First Principle

Before starting any non-trivial task, check if a relevant skill exists:

1. Identify task keywords from user message
2. Call `skill_search("<keywords>")` tool
3. If matches found:
   - Read the SKILL.md content (use `read_file`)
   - Follow its workflow/guidance for the current task
4. If no matches: proceed with general capabilities

**Skills are your extended capability** — don't reinvent what a skill already provides.

Common trigger keywords:

- frontend / UI / chart / diagram → creative skills
- github / PR / code review → software-development skills
- debug / TDD / refactor → core skills
- research / arxiv / osint → research skills
