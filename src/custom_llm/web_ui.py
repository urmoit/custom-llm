from __future__ import annotations

import argparse
import importlib
import html
import json
import os
import sys
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict
from urllib.parse import urlparse

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from .assistant import SmartAssistant

HOST_DEFAULT = "127.0.0.1"
PORT_DEFAULT = 8787


def _json_response(handler: BaseHTTPRequestHandler, payload: Dict[str, Any], status: int = 200) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(body)


def _read_json(handler: BaseHTTPRequestHandler) -> Dict[str, Any]:
    content_length = int(handler.headers.get("Content-Length", "0") or "0")
    raw = handler.rfile.read(content_length) if content_length else b""
    if not raw:
        return {}
    try:
        data = json.loads(raw.decode("utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}
def _current_versions() -> tuple[str, str, str]:
    version_module = importlib.import_module("custom_llm.version")
    version_module = importlib.reload(version_module)
    web_version = str(getattr(version_module, "WEB_UI_VERSION", "0.0.0"))
    cli_version = str(getattr(version_module, "CLI_UI_VERSION", version_module.VERSION))
    llm_version = str(getattr(version_module, "LLM_MODEL_VERSION", cli_version))
    return web_version, cli_version, llm_version


def _command_specs() -> Dict[str, str]:
    return {
        "help": "Show commands",
        "version": "Show web, CLI, and model versions",
        "model_info": "Show custom LLM architecture and stats",
        "gpu_status": "Show CUDA and training backend status",
        "search": "Force web search",
        "retrain": "Learn from conversations and retrain the model",
        "refresh": "Refresh chat session and reload the page",
        "clear": "Clear screen",
        "exit": "Quit",
    }


def _gpu_status_text() -> str:
    lines = []

    cuda_path = os.environ.get("CUDA_PATH", "")
    if cuda_path:
        lines.append(f"CUDA_PATH: {cuda_path}")
        nvcc = os.path.join(cuda_path, "bin", "nvcc.exe")
        lines.append(f"nvcc in CUDA_PATH: {'yes' if os.path.exists(nvcc) else 'no'}")
    else:
        lines.append("CUDA_PATH: not set")

    try:
        import torch  # type: ignore

        cuda_ok = bool(torch.cuda.is_available())
        torch_ver = str(torch.__version__)
        cuda_ver = str(torch.version.cuda or "")
        target_ok = torch_ver.startswith("2.9") and cuda_ver.startswith("13.") and cuda_ok

        lines.append(f"PyTorch installed: yes (version {torch_ver})")
        lines.append(f"CUDA available to PyTorch: {'yes' if cuda_ok else 'no'}")
        lines.append(f"CUDA runtime version: {torch.version.cuda or 'unknown'}")
        lines.append(
            "Target (torch 2.9.x + CUDA 13.x + cuda_available): "
            + ("PASS" if target_ok else "FAIL")
        )
        if cuda_ok:
            count = int(torch.cuda.device_count())
            lines.append(f"GPU count: {count}")
            if count > 0:
                lines.append(f"Primary GPU: {torch.cuda.get_device_name(0)}")
        else:
            lines.append("GPU note: CUDA is not available to this Python environment.")
    except Exception as exc:
        lines.append(f"PyTorch/CUDA check failed: {exc}")

    return "\n".join(lines)


def _format_version_summary() -> str:
    web_version, cli_version, llm_version = _current_versions()
    return f"Web UI v{web_version} | CLI UI v{cli_version} | LLM model v{llm_version}"


def _model_info_dict() -> Dict[str, Any]:
    """Return a JSON-serialisable dict of custom LLM metadata from the saved artifact."""
    from .config import MODEL_META_FILE
    if not MODEL_META_FILE.exists():
        return {"backend": "none", "status": "No model trained yet. Run build_info_and_train.bat."}
    try:
        meta = json.loads(MODEL_META_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {"backend": "unknown", "status": "Could not read model metadata."}

    backend = meta.get("backend", "unknown")
    result: Dict[str, Any] = {
        "backend": backend,
        "model_name": meta.get("model_name", "unknown"),
        "device": meta.get("device", "cpu"),
        "corpus_size": meta.get("corpus_size", 0),
    }
    if backend == "custom":
        result["num_parameters"] = meta.get("num_parameters", 0)
        result["vocab_size"] = meta.get("vocab_size", 0)
        result["d_model"] = meta.get("d_model", 256)
        result["n_layers"] = meta.get("n_layers", 4)
        result["n_heads"] = meta.get("n_heads", 4)
        result["context_length"] = meta.get("context_length", 256)
        result["epochs"] = meta.get("epochs", 0)
        result["status"] = "Custom transformer LLM (trained from scratch)"
    elif backend == "tfidf":
        result["vocab_size"] = meta.get("vocab_size", 0)
        result["status"] = "TF-IDF retrieval (no generation)"
    else:
        result["status"] = meta.get("note", "")
    return result


def _html_page() -> str:
    version_text = html.escape(_format_version_summary())
    info = _model_info_dict()
    backend_label = html.escape(str(info.get("backend", "unknown")).upper())
    params_val = info.get("num_parameters", 0)
    params_text = html.escape(f"{int(params_val):,}" if params_val else "—")
    vocab_val = info.get("vocab_size", 0)
    vocab_text = html.escape(f"{int(vocab_val):,}" if vocab_val else "—")
    corpus_val = info.get("corpus_size", 0)
    corpus_text = html.escape(f"{int(corpus_val):,}" if corpus_val else "—")
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Custom LLM Web UI</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #07111f;
      --panel: rgba(12, 20, 36, 0.86);
      --panel-2: rgba(18, 28, 48, 0.92);
      --border: rgba(148, 163, 184, 0.18);
      --text: #e5eefb;
      --muted: #94a3b8;
      --accent: #66e3c4;
      --accent-2: #7da8ff;
      --user: #142340;
      --bot: #112f2a;
      --shadow: 0 20px 60px rgba(0, 0, 0, 0.35);
    }}

    * {{ box-sizing: border-box; }}

    body {{
      margin: 0;
      min-height: 100vh;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle at top left, rgba(125, 168, 255, 0.24), transparent 28%),
        radial-gradient(circle at top right, rgba(102, 227, 196, 0.16), transparent 26%),
        linear-gradient(180deg, #07111f 0%, #050b15 100%);
    }}

    .shell {{
      max-width: 1100px;
      margin: 0 auto;
      padding: 28px 18px 20px;
    }}

    .hero {{
      display: grid;
      grid-template-columns: 1.35fr 0.9fr;
      gap: 16px;
      align-items: stretch;
      margin-bottom: 16px;
    }}

    .card {{
      border: 1px solid var(--border);
      border-radius: 22px;
      background: var(--panel);
      box-shadow: var(--shadow);
      backdrop-filter: blur(18px);
    }}

    .brand {{
      padding: 22px 22px 18px;
    }}

    .kicker {{
      color: var(--accent);
      text-transform: uppercase;
      letter-spacing: 0.16em;
      font-size: 12px;
      font-weight: 700;
      margin-bottom: 10px;
    }}

    h1 {{
      margin: 0;
      font-size: clamp(30px, 4vw, 54px);
      line-height: 1.03;
      letter-spacing: -0.04em;
    }}

    .sub {{
      margin: 14px 0 0;
      color: var(--muted);
      max-width: 62ch;
      line-height: 1.6;
    }}

    .stats {{
      display: grid;
      gap: 12px;
      padding: 18px;
    }}

    .stat {{
      padding: 16px;
      border-radius: 18px;
      background: var(--panel-2);
      border: 1px solid var(--border);
    }}

    .stat .label {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.12em; }}
    .stat .value {{ margin-top: 6px; font-size: 16px; font-weight: 600; }}

    .layout {{
      display: grid;
      grid-template-columns: 300px 1fr;
      gap: 16px;
      min-height: 72vh;
    }}

    .sidebar {{ padding: 18px; }}
    .sidebar h2 {{ margin: 0 0 12px; font-size: 18px; }}
    .sidebar p {{ margin: 0 0 16px; color: var(--muted); line-height: 1.6; }}
    .sidebar-section {{ margin-top: 18px; }}
    .chips {{ display: flex; flex-wrap: wrap; gap: 10px; }}
    .chip {{
      border: 1px solid var(--border);
      background: rgba(255, 255, 255, 0.03);
      color: var(--text);
      border-radius: 999px;
      padding: 10px 12px;
      cursor: pointer;
      transition: transform 0.15s ease, border-color 0.15s ease, background 0.15s ease;
    }}
    .chip:hover {{ transform: translateY(-1px); border-color: rgba(102, 227, 196, 0.45); background: rgba(102, 227, 196, 0.08); }}
    .command-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }}
    .command-btn {{
      border: 1px solid var(--border);
      background: rgba(255, 255, 255, 0.03);
      color: var(--text);
      border-radius: 14px;
      padding: 11px 12px;
      cursor: pointer;
      transition: transform 0.15s ease, border-color 0.15s ease, background 0.15s ease;
      text-align: left;
      font-size: 13px;
    }}
    .command-btn:hover {{ transform: translateY(-1px); border-color: rgba(125, 168, 255, 0.45); background: rgba(125, 168, 255, 0.08); }}
    .command-btn.retrain-btn {{ border-color: rgba(102, 227, 196, 0.28); }}
    .command-btn.retrain-btn:hover {{ border-color: rgba(102, 227, 196, 0.65); background: rgba(102, 227, 196, 0.1); }}
    .command-btn .cmd {{ display: block; font-weight: 700; color: var(--accent); margin-bottom: 4px; font-family: ui-monospace, SFMono-Regular, Consolas, Menlo, monospace; }}
    .command-btn .desc {{ color: var(--muted); line-height: 1.35; }}

    .chat {{ display: flex; flex-direction: column; overflow: hidden; }}
    .messages {{
      flex: 1;
      padding: 18px;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 14px;
      scroll-behavior: smooth;
    }}

    .message {{
      max-width: min(760px, 92%);
      padding: 14px 16px;
      border-radius: 18px;
      border: 1px solid var(--border);
      line-height: 1.55;
      word-wrap: break-word;
    }}
    .message.user {{ align-self: flex-end; background: var(--user); border-top-right-radius: 8px; }}
    .message.bot {{ align-self: flex-start; background: var(--bot); border-top-left-radius: 8px; }}
    .message.bot.script {{ max-width: min(900px, 96%); }}
    .message-body {{ white-space: pre-wrap; }}
    .meta {{ margin-top: 8px; color: var(--muted); font-size: 12px; }}
    .script-intro {{ margin-bottom: 14px; white-space: pre-wrap; }}
    .script-usage {{ margin-top: 12px; color: var(--muted); white-space: pre-wrap; }}
    .code-card {{
      position: relative;
      border: 1px solid rgba(125, 168, 255, 0.24);
      border-radius: 16px;
      background: rgba(2, 7, 18, 0.95);
      overflow: hidden;
      box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.03);
    }}
    .code-head {{
      display: flex;
      justify-content: flex-end;
      padding: 10px 10px 0;
    }}
    .copy-btn {{
      border: 1px solid rgba(148, 163, 184, 0.22);
      background: rgba(255, 255, 255, 0.04);
      color: var(--text);
      border-radius: 10px;
      padding: 6px 10px;
      cursor: pointer;
      font-size: 12px;
      line-height: 1;
    }}
    .copy-btn:hover {{ background: rgba(255, 255, 255, 0.08); }}
    .code-card pre {{
      margin: 0;
      padding: 12px 16px 16px;
      color: #d7e8ff;
      font-size: 13px;
      line-height: 1.55;
      overflow-x: auto;
      font-family: ui-monospace, SFMono-Regular, Consolas, Menlo, Monaco, monospace;
      white-space: pre;
    }}

    .composer {{
      padding: 16px;
      border-top: 1px solid var(--border);
      background: rgba(4, 10, 18, 0.78);
    }}

    .row {{ display: grid; grid-template-columns: 1fr auto; gap: 12px; align-items: end; }}
    textarea {{
      width: 100%;
      min-height: 58px;
      max-height: 180px;
      resize: vertical;
      padding: 14px 15px;
      border-radius: 16px;
      border: 1px solid var(--border);
      background: rgba(255, 255, 255, 0.03);
      color: var(--text);
      font: inherit;
      outline: none;
    }}
    textarea:focus {{ border-color: rgba(102, 227, 196, 0.5); box-shadow: 0 0 0 4px rgba(102, 227, 196, 0.08); }}

    .actions {{ display: flex; gap: 10px; flex-wrap: wrap; }}
    button {{
      border: 0;
      border-radius: 14px;
      padding: 14px 16px;
      font: inherit;
      font-weight: 700;
      cursor: pointer;
      transition: transform 0.15s ease, opacity 0.15s ease;
    }}
    button:hover {{ transform: translateY(-1px); }}
    .primary {{ background: linear-gradient(135deg, var(--accent), var(--accent-2)); color: #04111d; }}
    .ghost {{ background: rgba(255, 255, 255, 0.06); color: var(--text); border: 1px solid var(--border); }}

    .status {{ margin-top: 10px; color: var(--muted); min-height: 20px; font-size: 13px; }}

    @media (max-width: 900px) {{
      .hero, .layout {{ grid-template-columns: 1fr; }}
      .shell {{ padding: 16px; }}
      .message {{ max-width: 100%; }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <div class="hero">
      <section class="card brand">
        <div class="kicker">Fully custom transformer AI — no pre-trained weights</div>
        <h1>Custom LLM Web UI</h1>
        <p class="sub">A local chat interface for your custom GPT-style transformer. Every weight was trained from scratch on your knowledge base. Chat history is remembered and used on the next retrain.</p>
      </section>
      <aside class="card stats">
        <div class="stat">
          <div class="label">Versions</div>
          <div class="value">{version_text}</div>
        </div>
        <div class="stat">
          <div class="label">Backend</div>
          <div class="value">{backend_label}</div>
        </div>
        <div class="stat">
          <div class="label">Parameters</div>
          <div class="value">{params_text}</div>
        </div>
        <div class="stat">
          <div class="label">Vocab size</div>
          <div class="value">{vocab_text}</div>
        </div>
        <div class="stat">
          <div class="label">Corpus documents</div>
          <div class="value">{corpus_text}</div>
        </div>
      </aside>
    </div>

    <div class="layout">
      <aside class="card sidebar">
        <h2>Quick prompts</h2>
        <p>Pick a starter prompt, or write your own. The browser UI talks to the same local assistant backend.</p>
        <div class="chips" id="chips">
          <button class="chip" data-prompt="what are ongoing policy challenges in usa">USA policy challenges</button>
          <button class="chip" data-prompt="generate python script to organize files by extension">Python file organizer</button>
          <button class="chip" data-prompt="what can you tell me about python scripting basics">Python basics</button>
          <button class="chip" data-prompt="hello">Greeting</button>
        </div>
        <div class="sidebar-section">
          <h2>Commands</h2>
          <p>These match the CLI commands. Click one to insert it into the box, then send it.</p>
          <div class="command-grid" id="commands">
            <button class="command-btn" data-command="/help"><span class="cmd">/help</span><span class="desc">Show commands</span></button>
            <button class="command-btn" data-command="/version"><span class="cmd">/version</span><span class="desc">Show versions</span></button>
            <button class="command-btn" data-command="/model_info"><span class="cmd">/model_info</span><span class="desc">Custom LLM stats</span></button>
            <button class="command-btn" data-command="/gpu_status"><span class="cmd">/gpu_status</span><span class="desc">Show GPU status</span></button>
            <button class="command-btn" data-command="/search "><span class="cmd">/search</span><span class="desc">Force web search</span></button>
            <button class="command-btn retrain-btn" data-command="/retrain"><span class="cmd">/retrain</span><span class="desc">Learn from chats + retrain</span></button>
            <button class="command-btn" data-command="/refresh"><span class="cmd">/refresh</span><span class="desc">Reload page + backend</span></button>
            <button class="command-btn" data-command="/clear"><span class="cmd">/clear</span><span class="desc">Clear chat</span></button>
            <button class="command-btn" data-command="/exit"><span class="cmd">/exit</span><span class="desc">Close the session</span></button>
          </div>
        </div>
        <div class="status" id="sidebarStatus">Ready.</div>
      </aside>

      <main class="card chat">
        <div class="messages" id="messages">
          <div class="message bot">
            Hello. Ask a question or try one of the quick prompts.
            <div class="meta">Local assistant ready</div>
          </div>
        </div>
        <div class="composer">
          <div class="row">
            <textarea id="input" placeholder="Type your message here..." rows="2"></textarea>
            <div class="actions">
              <button class="primary" id="sendBtn">Send</button>
              <button class="ghost" id="clearBtn">Clear</button>
            </div>
          </div>
          <div class="status" id="status">Type a message and press Enter to send.</div>
        </div>
      </main>
    </div>
  </div>

  <script>
    const messages = document.getElementById('messages');
    const input = document.getElementById('input');
    const status = document.getElementById('status');
    const sidebarStatus = document.getElementById('sidebarStatus');
    const sendBtn = document.getElementById('sendBtn');
    const clearBtn = document.getElementById('clearBtn');
    const chips = document.getElementById('chips');
    const commands = document.getElementById('commands');

    const COMMAND_LINES = [
      '/help        Show commands',
      '/version     Show web, CLI, and model versions',
      '/model_info  Show custom LLM architecture and stats',
      '/gpu_status  Show CUDA and training backend status',
      '/search      Force web search',
      '/retrain     Learn from your conversations and retrain the model',
      '/refresh     Refresh chat session and reload the page',
      '/clear       Clear screen',
      '/exit        Quit',
    ];

    function scrollToBottom() {{
      messages.scrollTop = messages.scrollHeight;
    }}

    function setStatus(text) {{
      status.textContent = text;
      sidebarStatus.textContent = text;
    }}

    function clearMessages(includeGreeting = true) {{
      messages.innerHTML = '';
      if (includeGreeting) {{
        addMessage(
          'Hello. Ask a question or try one of the quick prompts.',
          'bot',
          'Local assistant ready',
          'plain'
        );
      }}
    }}

    function makeMetaNode(meta) {{
      if (!meta) return null;
      const metaNode = document.createElement('div');
      metaNode.className = 'meta';
      metaNode.textContent = meta;
      return metaNode;
    }}

    function createCopyButton(code) {{
      const button = document.createElement('button');
      button.className = 'copy-btn';
      button.type = 'button';
      button.textContent = 'Copy';
      button.addEventListener('click', async () => {{
        try {{
          await navigator.clipboard.writeText(code);
          button.textContent = 'Copied';
          setTimeout(() => {{ button.textContent = 'Copy'; }}, 1400);
        }} catch (error) {{
          button.textContent = 'Copy failed';
          setTimeout(() => {{ button.textContent = 'Copy'; }}, 1400);
        }}
      }});
      return button;
    }}

    function splitScriptAnswer(text) {{
      const usageMarker = '\\nUsage:';
      let usage = '';
      let body = text.trim();

      const usageIndex = body.indexOf(usageMarker);
      if (usageIndex >= 0) {{
        usage = body.slice(usageIndex + 1).trim();
        body = body.slice(0, usageIndex).trimEnd();
      }}

      const separator = '\\n\\n';
      const introBreak = body.indexOf(separator);
      if (introBreak < 0) {{
        return {{ intro: '', code: body, usage }};
      }}

      const intro = body.slice(0, introBreak).trim();
      const code = body.slice(introBreak + separator.length).trim();
      return {{ intro, code, usage }};
    }}

    function createCodeCard(code) {{
      const card = document.createElement('div');
      card.className = 'code-card';

      const head = document.createElement('div');
      head.className = 'code-head';
      head.appendChild(createCopyButton(code));

      const pre = document.createElement('pre');
      pre.textContent = code;

      card.appendChild(head);
      card.appendChild(pre);
      return card;
    }}

    function createPlainBody(text) {{
      const body = document.createElement('div');
      body.className = 'message-body';
      body.textContent = text;
      return body;
    }}

    function createScriptBody(text) {{
      const parts = splitScriptAnswer(text);
      const wrapper = document.createElement('div');

      if (parts.intro) {{
        const intro = document.createElement('div');
        intro.className = 'script-intro';
        intro.textContent = parts.intro;
        wrapper.appendChild(intro);
      }}

      if (parts.code) {{
        wrapper.appendChild(createCodeCard(parts.code));
      }}

      if (parts.usage) {{
        const usage = document.createElement('div');
        usage.className = 'script-usage';
        usage.textContent = parts.usage;
        wrapper.appendChild(usage);
      }}

      return wrapper;
    }}

    function addMessage(text, kind, meta, source) {{
      const node = document.createElement('div');
      node.className = 'message ' + kind;

      if (kind === 'bot' && source === 'python-script-generator') {{
        node.classList.add('script');
        node.appendChild(createScriptBody(text));
      }} else {{
        node.appendChild(createPlainBody(text));
      }}

      const metaNode = makeMetaNode(meta);
      if (metaNode) {{
        node.appendChild(metaNode);
      }}

      messages.appendChild(node);
      scrollToBottom();
    }}

    function showCommandList() {{
      addMessage(COMMAND_LINES.join('\\n'), 'bot', 'CLI command list', 'plain');
    }}

    async function fetchJson(url, options) {{
      const response = await fetch(url, options);
      const data = await response.json();
      if (!response.ok) {{
        throw new Error(data.error || data.message || 'Request failed');
      }}
      return data;
    }}

    async function sendChat(prompt, forceWeb = false) {{
      const message = prompt.trim();
      if (!message) {{
        return;
      }}

      addMessage(message, 'user');
      input.value = '';
      input.focus();
      setStatus('Thinking...');

      try {{
        const data = await fetchJson('/api/chat', {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/json' }},
          body: JSON.stringify({{ message, force_web: forceWeb }})
        }});
        const meta = 'source=' + data.source + ' | confidence=' + (data.confidence ?? 'n/a');
        addMessage(data.answer, 'bot', meta, data.source);
        setStatus('Ready.');
      }} catch (error) {{
        addMessage('Error: ' + error.message, 'bot', 'request failed', 'plain');
        setStatus('Request failed.');
      }}
    }}

    async function showVersion() {{
      try {{
        const data = await fetchJson('/api/meta');
        addMessage(
          'Web UI version: ' + data.web_ui_version + '\\nCLI UI version: ' + data.cli_version + '\\nLLM model version: ' + data.llm_version,
          'bot',
          'version info',
          'plain'
        );
        setStatus('Ready.');
      }} catch (error) {{
        addMessage('Error: ' + error.message, 'bot', 'request failed', 'plain');
        setStatus('Request failed.');
      }}
    }}

    async function showGpuStatus() {{
      try {{
        const data = await fetchJson('/api/gpu_status');
        addMessage(data.status, 'bot', 'system status', 'plain');
        setStatus('Ready.');
      }} catch (error) {{
        addMessage('Error: ' + error.message, 'bot', 'request failed', 'plain');
        setStatus('Request failed.');
      }}
    }}

    async function showModelInfo() {{
      try {{
        const data = await fetchJson('/api/model_info');
        const lines = [
          'Custom LLM Architecture',
          '==============================',
          'Backend: ' + (data.backend || 'unknown').toUpperCase(),
          'Model: ' + (data.model_name || 'unknown'),
          'Device: ' + (data.device || 'cpu'),
        ];
        if (data.num_parameters) lines.push('Parameters: ' + Number(data.num_parameters).toLocaleString());
        if (data.vocab_size) lines.push('Vocabulary size: ' + Number(data.vocab_size).toLocaleString());
        if (data.d_model) lines.push('d_model: ' + data.d_model);
        if (data.n_layers) lines.push('n_layers: ' + data.n_layers);
        if (data.n_heads) lines.push('n_heads: ' + data.n_heads);
        if (data.context_length) lines.push('context_length: ' + data.context_length);
        if (data.corpus_size) lines.push('Corpus documents: ' + Number(data.corpus_size).toLocaleString());
        if (data.epochs) lines.push('Training epochs: ' + data.epochs);
        lines.push('');
        lines.push(data.status || '');
        lines.push('');
        lines.push('Use /retrain to rebuild the model including your chat history.');
        addMessage(lines.join('\\n'), 'bot', 'model info', 'plain');
        setStatus('Ready.');
      }} catch (error) {{
        addMessage('Error: ' + error.message, 'bot', 'request failed', 'plain');
        setStatus('Request failed.');
      }}
    }}

    async function retrainModel() {{
      setStatus('Retraining... this may take a minute.');
      addMessage(
        'Starting retrain — learning from all knowledge files and your conversation history. Please wait...',
        'bot', 'system', 'plain'
      );
      try {{
        const data = await fetchJson('/api/retrain', {{ method: 'POST' }});
        addMessage(data.message || 'Retrain complete.', 'bot', 'retrain result', 'plain');
        setStatus('Retrain complete. Reload the page to see updated stats.');
      }} catch (error) {{
        addMessage('Retrain failed: ' + error.message, 'bot', 'error', 'plain');
        setStatus('Retrain failed.');
      }}
    }}

    async function refreshUi() {{
      setStatus('Refreshing...');
      try {{
        await fetchJson('/api/refresh', {{ method: 'POST' }});
        window.location.reload();
      }} catch (error) {{
        addMessage('Error: ' + error.message, 'bot', 'request failed', 'plain');
        setStatus('Refresh failed.');
      }}
    }}

    async function handleSlashCommand(rawValue) {{
      const cmdline = rawValue.trim().replace(/^\\/+/, '');
      if (!cmdline) {{
        showCommandList();
        return true;
      }}

      const parts = cmdline.split(/\\s+/, 2);
      const cmd = parts[0].toLowerCase().replace(/-/g, '_');
      const arg = parts.length > 1 ? parts[1].trim() : '';

      if (cmd === 'help') {{
        showCommandList();
        return true;
      }}
      if (cmd === 'version') {{
        await showVersion();
        return true;
      }}
      if (cmd === 'gpu' || cmd === 'gpu_status') {{
        await showGpuStatus();
        return true;
      }}
      if (cmd === 'model_info') {{
        await showModelInfo();
        return true;
      }}
      if (cmd === 'retrain') {{
        await retrainModel();
        return true;
      }}
      if (cmd === 'search') {{
        if (!arg) {{
          addMessage('Usage: /search <query>', 'bot', 'system', 'plain');
          return true;
        }}
        await sendChat(arg, true);
        return true;
      }}
      if (cmd === 'refresh') {{
        await refreshUi();
        return true;
      }}
      if (cmd === 'clear') {{
        clearMessages(true);
        setStatus('Ready.');
        return true;
      }}
      if (cmd === 'exit') {{
        addMessage('Close this browser tab to exit.', 'bot', 'system', 'plain');
        return true;
      }}

      addMessage('Unknown command. Type /help for the list.', 'bot', 'system', 'plain');
      return true;
    }}

    async function submitInput() {{
      const value = input.value.trim();
      if (!value) {{
        return;
      }}

      if (value.startsWith('/')) {{
        input.value = '';
        input.focus();
        await handleSlashCommand(value);
        return;
      }}

      await sendChat(value, false);
    }}

    sendBtn.addEventListener('click', submitInput);
    clearBtn.addEventListener('click', () => {{
      clearMessages(true);
      setStatus('Ready.');
      input.focus();
    }});

    chips.addEventListener('click', (event) => {{
      const button = event.target.closest('button[data-prompt]');
      if (!button) return;
      input.value = button.dataset.prompt;
      input.focus();
      setStatus('Prompt loaded. Press Send.');
    }});

    commands.addEventListener('click', (event) => {{
      const button = event.target.closest('button[data-command]');
      if (!button) return;
      input.value = button.dataset.command;
      input.focus();
      if (button.dataset.command === '/search ') {{
        setStatus('Type a search query after /search.');
      }} else {{
        setStatus('Command loaded. Press Send.');
      }}
    }});

    input.addEventListener('keydown', (event) => {{
      if (event.key === 'Enter' && !event.shiftKey) {{
        event.preventDefault();
        submitInput();
      }}
    }});

    clearMessages(true);
    setStatus('Type a message or use a command.');
  </script>
</body>
</html>"""


class _WebUIHandler(BaseHTTPRequestHandler):
    assistant = SmartAssistant()

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path in {"/", "/index.html"}:
            page = _html_page().encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(page)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(page)
            return

        if parsed.path == "/api/meta":
            web_version, cli_version, llm_version = _current_versions()
            _json_response(
                self,
                {
                    "web_ui_version": web_version,
                    "cli_version": cli_version,
                    "llm_version": llm_version,
                    "title": "Custom LLM Web UI",
                },
            )
            return

        if parsed.path == "/api/gpu_status":
            _json_response(self, {"status": _gpu_status_text()})
            return

        if parsed.path == "/api/model_info":
            _json_response(self, _model_info_dict())
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/api/chat":
            data = _read_json(self)
            message = str(data.get("message", "")).strip()
            if not message:
                _json_response(self, {"error": "Missing message"}, status=HTTPStatus.BAD_REQUEST)
                return

            force_web = bool(data.get("force_web", False))
            result = self.assistant.ask(message, force_web=force_web)
            _json_response(
                self,
                {
                    "answer": result.answer,
                    "source": result.source,
                    "thinking": result.thinking,
                    "confidence": result.confidence,
                },
            )
            return

        if parsed.path == "/api/refresh":
            type(self).assistant = SmartAssistant()
            web_version, cli_version, llm_version = _current_versions()
            _json_response(
                self,
                {
                    "message": "Web UI refreshed.",
                    "web_ui_version": web_version,
                    "cli_version": cli_version,
                    "llm_version": llm_version,
                },
            )
            return

        if parsed.path == "/api/retrain":
            try:
                msg = type(self).assistant.retrain_and_reload()
                _json_response(self, {"message": msg, "status": "ok"})
            except Exception as exc:
                _json_response(
                    self,
                    {"error": str(exc), "status": "error"},
                    status=HTTPStatus.INTERNAL_SERVER_ERROR,
                )
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Not found")


def create_server(host: str, port: int) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), _WebUIHandler)


def run_web_ui(host: str = HOST_DEFAULT, port: int = PORT_DEFAULT, open_browser: bool = False) -> None:
    server = create_server(host, port)
    url = f"http://{host}:{port}"
    print(f"[web-ui] Serving on {url}")
    if open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("[web-ui] Stopping.")
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Custom LLM browser UI")
    parser.add_argument("--host", default=HOST_DEFAULT)
    parser.add_argument("--port", default=PORT_DEFAULT, type=int)
    parser.add_argument("--open-browser", action="store_true")
    args = parser.parse_args()
    run_web_ui(host=args.host, port=args.port, open_browser=args.open_browser)


if __name__ == "__main__":
    main()
