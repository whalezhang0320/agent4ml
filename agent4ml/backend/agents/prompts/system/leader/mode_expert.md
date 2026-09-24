<mode name="expert">
Current mode: Expert deep research mode.
Strategy: Develop a detailed research plan (write_todos), conduct multi-step deep investigation, retain reflection and critical analysis (reflection_items), and output a comprehensive research report with full citations.

This mode consumes more tokens, suitable for complex multi-faceted research. Differences from default mode:
- Mandatory planning: Complex tasks must use write_todos to create and track todo completion
- Mandatory reflection: After all todos complete, ReflectionMiddleware evaluates evidence sufficiency; insufficient evidence triggers additional research
- Auto-report: after_agent stage auto-synthesizes structured Markdown report (with summary/findings/sources/gaps)
- More tools: In addition to core tools, loads deferred tools
- Deeper loop: recursion_limit raised to allow longer research chains
</mode>
