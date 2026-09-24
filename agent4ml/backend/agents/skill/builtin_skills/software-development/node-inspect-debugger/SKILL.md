---
name: node-inspect-debugger
description: "Debug Node.js via --inspect + Chrome DevTools Protocol."
allowed-tools:
  - bash
  - read_file
enabled: true
related-skills: [systematic-debugging, python-debugpy]
license: MIT
author: Adapted from hermes-agent (Nous Research, MIT)
---

# Node.js Inspect Debugger

## Overview

When `console.log` isn't enough, drive Node's built-in V8 inspector
programmatically from the terminal. Real breakpoints, step in/over/out,
call-stack walking, scope dumps, and arbitrary expression evaluation.

**Prefer `node inspect` first.** It's always available and the REPL is fast.

## When to Use

- A Node test fails and you need to see intermediate state
- A Node process crashes or behaves wrong and you want to inspect state
- You need to inspect a value in a closure that `console.log` can't reach
- Perf: attach to a running process to capture a CPU profile or heap snapshot

**Don't use for:** things `console.log` solves in under a minute.

## Quick Reference: `node inspect` REPL

Launch paused on first line:

```bash
node inspect path/to/script.js
# or with tsx
node --inspect-brk $(which tsx) path/to/script.ts
```

The `debug>` prompt accepts:

| Command | Action |
|---------|--------|
| `c` | continue |
| `n` | next line (step over) |
| `s` | step into |
| `o` | step out |
| `sb('file.js', line)` | set breakpoint |
| `cb('file.js', line)` | clear breakpoint |
| `watch('expr')` | watch expression |
| `exec('expr')` | evaluate in paused frame |
| `backtrace` / `bt` | show call stack |
| `list(N)` | show N lines around current |

## Scriptable CDP via chrome-remote-interface

For automated debugging (many breakpoints, collect state across runs):

```bash
# Install
npm install chrome-remote-interface

# Launch node with inspect
node --inspect-brk=9229 path/to/script.js &

# Attach via script
node -e "
const CDP = require('chrome-remote-interface');
(async () => {
  const client = await CDP({port: 9229});
  const {Debugger, Runtime} = client;
  await Debugger.enable();
  await Runtime.enable();
  Debugger.paused(({callFrames}) => {
    console.log('Paused at:', callFrames[0].location);
    // Inspect scope, evaluate expressions, etc.
  });
  // Set breakpoint
  await Debugger.setBreakpointByUrl({lineNumber: 10, url: 'file:///path/to/script.js'});
  await Debugger.resume();
})();
"
```

## Attaching to a Running Process

```bash
# Start with --inspect (no brk = doesn't pause on start)
node --inspect=9229 path/to/server.js

# Attach from another terminal
node inspect localhost:9229
```

## CPU Profile

```bash
# Start with inspect
node --inspect path/to/script.js

# In another terminal, capture profile
node -e "
const CDP = require('chrome-remote-interface');
(async () => {
  const client = await CDP({port: 9229});
  const {Profiler} = client;
  await Profiler.enable();
  await Profiler.start();
  // Wait for workload...
  await new Promise(r => setTimeout(r, 10000));
  const {profile} = await Profiler.stop();
  require('fs').writeFileSync('profile.cpuprofile', JSON.stringify(profile));
})();
"

# Load profile.cpuprofile in Chrome DevTools > Performance
```

## Pitfalls

- **--inspect vs --inspect-brk**: `--inspect` doesn't pause; `--inspect-brk`
  pauses on first line. Use `brk` when you need to catch early code.
- **Port conflicts**: default port 9229 may be taken. Use `--inspect=9230`.
- **ESM vs CJS**: `node inspect` works with both, but ESM may show different
  call stack formatting.
- **Source maps**: TypeScript/tsx may show transpiled line numbers. Use
  `--enable-source-maps` for accurate mapping.
