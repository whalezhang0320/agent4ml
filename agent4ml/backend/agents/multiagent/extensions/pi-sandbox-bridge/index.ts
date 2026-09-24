/**
 * pi-sandbox-bridge — Pi extension 转发 8 工具到 Agent4ML SpecialistMcpServer.
 *
 * 决策 1（设计文档 46 §10.5）：强制走 Agent4ML SpecialistMcpServer。
 * pi --no-builtin-tools 禁用自带 read/write/edit/bash，
 * 此 extension 注册 8 个 agent4ml_* 工具，全部通过 stdio MCP 转发到
 * Agent4ML SpecialistMcpServer（AGENT4ML_SANDBOX_MCP_ENDPOINT env 指定）。
 *
 * 经过 PathTranslator + SecurityGuard（与 codex/claude 统一沙箱访问）。
 * 物理无隔离，pi 写的文件 lead agent 立即可见。
 *
 * 注意：此文件放在 Python 仓库（Agent4ML）内，但运行时由 pi 用 jiti 加载。
 * jiti 跳过 tsc 类型检查，所以 // @ts-nocheck 不影响运行时。
 * 去掉 typebox 依赖（pi.registerTool 接受 plain JSON Schema object）。
 */

// @ts-nocheck — pi 用 jiti 加载，跳过 tsc 类型检查；Agent4ML 仓库无 npm 依赖

// 8 个工具名 + 参数 schema（与 Agent4ML SpecialistMcpServer 暴露的接口一致）
// 用 plain JSON Schema object，不依赖 @sinclair/typebox
const AGENT4ML_TOOLS = [
  {
    name: "bash",
    description: "Execute a bash command in the Agent4ML sandbox (security-guarded).",
    parameters: {
      type: "object",
      properties: {
        command: { type: "string", description: "The bash command to execute." },
      },
      required: ["command"],
    },
  },
  {
    name: "read_file",
    description: "Read a file from the Agent4ML sandbox (security-guarded).",
    parameters: {
      type: "object",
      properties: {
        path: { type: "string", description: "File path to read." },
      },
      required: ["path"],
    },
  },
  {
    name: "write_file",
    description: "Write content to a file in the Agent4ML sandbox (security-guarded).",
    parameters: {
      type: "object",
      properties: {
        path: { type: "string", description: "File path to write." },
        content: { type: "string", description: "Content to write." },
      },
      required: ["path", "content"],
    },
  },
  {
    name: "list_dir",
    description: "List directory contents in the Agent4ML sandbox (security-guarded).",
    parameters: {
      type: "object",
      properties: {
        path: { type: "string", description: "Directory path to list." },
      },
      required: ["path"],
    },
  },
  {
    name: "str_replace",
    description: "Replace string in a file in the Agent4ML sandbox (security-guarded).",
    parameters: {
      type: "object",
      properties: {
        path: { type: "string", description: "File path." },
        old_str: { type: "string", description: "String to replace." },
        new_str: { type: "string", description: "Replacement string." },
      },
      required: ["path", "old_str", "new_str"],
    },
  },
  {
    name: "glob",
    description: "Find files matching glob pattern in the Agent4ML sandbox (security-guarded).",
    parameters: {
      type: "object",
      properties: {
        pattern: { type: "string", description: "Glob pattern." },
        path: { type: "string", description: "Base path (optional)." },
      },
      required: ["pattern"],
    },
  },
  {
    name: "grep",
    description: "Search file contents with regex in the Agent4ML sandbox (security-guarded).",
    parameters: {
      type: "object",
      properties: {
        pattern: { type: "string", description: "Regex pattern." },
        path: { type: "string", description: "File or directory path." },
      },
      required: ["pattern", "path"],
    },
  },
  {
    name: "download_file",
    description: "Download a file to the Agent4ML sandbox (security-guarded).",
    parameters: {
      type: "object",
      properties: {
        url: { type: "string", description: "URL to download." },
        path: { type: "string", description: "Destination path." },
      },
      required: ["url", "path"],
    },
  },
];

/**
 * Extension entry point.
 *
 * pi 加载此 extension 后，agent 看到 agent4ml_bash / agent4ml_read_file / agent4ml_write_file 等 8 个工具。
 * 所有工具通过 stdio MCP 协议调 Agent4ML SpecialistMcpServer。
 */
export default function (pi) {
  for (const toolDef of AGENT4ML_TOOLS) {
    const toolName = toolDef.name;
    const agent4mlToolName = `agent4ml_${toolName}`;
    pi.registerTool({
      name: agent4mlToolName,
      label: `Agent4ML ${toolName}`,
      description: toolDef.description,
      parameters: toolDef.parameters,
      execute: async (_toolCallId, params) => {
        const result = await callAgent4MLMcp(toolName, params);
        return {
          content: [{ type: "text", text: result }],
          details: {},
        };
      },
    });
  }
}

/**
 * 通过 stdio MCP 协议调 Agent4ML SpecialistMcpServer.
 *
 * AGENT4ML_SANDBOX_MCP_ENDPOINT env 由 PiRuntime._build_env 注入，
 * 格式：python -m agent4ml.backend.agents.multiagent.mcp.specialist_mcp_server --sandbox-id <id>
 *
 * 此函数启动 SpecialistMcpServer 子进程，通过 stdio MCP JSON-RPC 调用指定 tool。
 *
 * MVP：每次调用启动新子进程（与 codex/claude 的 SpecialistMcpServer 调用模式一致）。
 * 进阶：可改为持久 MCP client 连接。
 */
async function callAgent4MLMcp(tool, args) {
  const endpoint = process.env.AGENT4ML_SANDBOX_MCP_ENDPOINT;
  if (!endpoint) {
    throw new Error(
      "AGENT4ML_SANDBOX_MCP_ENDPOINT not set. PiRuntime._build_env should inject it."
    );
  }

  // 解析 endpoint：python -m agent4ml.backend.agents.multiagent.mcp.specialist_mcp_server --sandbox-id <id>
  const cmdParts = endpoint.split(" ");

  // 启动 SpecialistMcpServer 子进程
  const { spawn } = require("child_process");
  const proc = spawn(cmdParts[0], cmdParts.slice(1), {
    stdio: ["pipe", "pipe", "pipe"],
  });

  return new Promise((resolve, reject) => {
    let stdoutBuffer = "";
    let stderrBuffer = "";

    proc.stdout.on("data", (data) => {
      stdoutBuffer += data.toString();
    });

    proc.stderr.on("data", (data) => {
      stderrBuffer += data.toString();
    });

    proc.on("error", (err) => {
      reject(new Error(`SpecialistMcpServer spawn failed: ${err.message}`));
    });

    proc.on("close", (code) => {
      if (code !== 0) {
        reject(
          new Error(
            `SpecialistMcpServer exited with code ${code}: ${stderrBuffer}`
          )
        );
        return;
      }
      // 尝试从 stdout 解析 MCP JSON-RPC response
      // MCP stdio 协议：每行一个 JSON-RPC message
      const lines = stdoutBuffer.split("\n").filter((l) => l.trim());
      for (const line of lines) {
        try {
          const msg = JSON.parse(line);
          // JSON-RPC response 含 result 或 error
          if (msg.result && msg.result.content) {
            // MCP tool result: { content: [{ type: "text", text: "..." }] }
            const textParts = msg.result.content
              .filter((c) => c.type === "text")
              .map((c) => c.text);
            resolve(textParts.join("\n") || "(empty result)");
            return;
          }
          if (msg.error) {
            reject(new Error(`MCP error: ${msg.error.message || JSON.stringify(msg.error)}`));
            return;
          }
        } catch {
          // 非 JSON 行跳过
          continue;
        }
      }
      // 无有效 MCP response，返原始 stdout
      resolve(stdoutBuffer || "(empty result)");
    });

    // 发送 MCP JSON-RPC 请求：initialize → callTool
    // MCP stdio 协议需先 initialize 再 callTool
    const initRequest = {
      jsonrpc: "2.0",
      id: 0,
      method: "initialize",
      params: {
        protocolVersion: "2024-11-05",
        capabilities: {},
        clientInfo: { name: "pi-sandbox-bridge", version: "0.1.0" },
      },
    };
    const callRequest = {
      jsonrpc: "2.0",
      id: 1,
      method: "tools/call",
      params: {
        name: tool,
        arguments: args,
      },
    };
    proc.stdin.write(JSON.stringify(initRequest) + "\n");
    proc.stdin.write(JSON.stringify(callRequest) + "\n");
    proc.stdin.end();
  });
}
