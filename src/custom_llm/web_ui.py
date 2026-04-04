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
    cmds_json = json.dumps([{"name": k, "desc": v} for k, v in _command_specs().items()])
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Custom LLM</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

    :root {{
      color-scheme: dark;
      --bg:          #212121;
      --sidebar-bg:  #171717;
      --input-bg:    #2f2f2f;
      --popup-bg:    #2a2a2a;
      --popup-hover: #383838;
      --border:      rgba(255,255,255,0.10);
      --text:        #ececec;
      --muted:       #8e8ea0;
      --accent:      #10a37f;
      --accent-blue: #7da8ff;
      --user-bg:     #2f2f2f;
      --shadow-lg:   0 8px 32px rgba(0,0,0,.55);
    }}

    html, body {{ height: 100%; overflow: hidden; }}

    body {{
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
                   "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
      font-size: 15px;
      line-height: 1.55;
    }}

    /* ── layout ──────────────────────────────────────────────────────── */
    .app {{
      display: flex;
      height: 100vh;
      overflow: hidden;
    }}

    /* ── sidebar ─────────────────────────────────────────────────────── */
    .sidebar {{
      width: 264px;
      flex-shrink: 0;
      background: var(--sidebar-bg);
      border-right: 1px solid var(--border);
      display: flex;
      flex-direction: column;
      overflow-y: auto;
      padding: 20px 16px 24px;
      gap: 24px;
    }}

    .brand {{
      display: flex;
      align-items: center;
      gap: 10px;
      padding-bottom: 4px;
    }}
    .brand-icon {{
      width: 34px; height: 34px;
      background: var(--accent);
      border-radius: 10px;
      display: flex; align-items: center; justify-content: center;
      font-size: 17px;
    }}
    .brand-title {{ font-size: 16px; font-weight: 700; }}
    .brand-sub {{ font-size: 11px; color: var(--muted); margin-top: 1px; }}

    .sidebar-section-title {{
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.09em;
      color: var(--muted);
      margin-bottom: 8px;
    }}

    .stat-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 6px;
    }}
    .stat-chip {{
      background: rgba(255,255,255,0.04);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 8px 10px;
    }}
    .stat-chip .s-label {{ font-size: 10px; color: var(--muted); text-transform: uppercase; letter-spacing: .06em; }}
    .stat-chip .s-val   {{ font-size: 13px; font-weight: 600; margin-top: 2px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
    .stat-chip.full {{ grid-column: 1 / -1; }}

    .prompt-list {{ display: flex; flex-direction: column; gap: 4px; }}
    .prompt-btn {{
      background: rgba(255,255,255,0.03);
      border: 1px solid var(--border);
      color: var(--text);
      text-align: left;
      padding: 9px 12px;
      border-radius: 10px;
      cursor: pointer;
      font-size: 13px;
      transition: background .15s, border-color .15s;
      white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    }}
    .prompt-btn:hover {{ background: rgba(255,255,255,0.07); border-color: rgba(255,255,255,0.2); }}

    .sidebar-status {{
      margin-top: auto;
      font-size: 12px;
      color: var(--muted);
      padding-top: 12px;
      border-top: 1px solid var(--border);
    }}

    /* ── main area ───────────────────────────────────────────────────── */
    .main {{
      flex: 1;
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }}

    /* ── topbar ──────────────────────────────────────────────────────── */
    .topbar {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 12px 20px;
      border-bottom: 1px solid var(--border);
      flex-shrink: 0;
      gap: 12px;
    }}
    .topbar-title {{ font-weight: 600; font-size: 14px; color: var(--muted); }}
    .topbar-actions {{ display: flex; gap: 8px; }}
    .topbar-btn {{
      background: rgba(255,255,255,0.05);
      border: 1px solid var(--border);
      color: var(--text);
      border-radius: 8px;
      padding: 6px 12px;
      font-size: 13px;
      cursor: pointer;
      transition: background .15s;
    }}
    .topbar-btn:hover {{ background: rgba(255,255,255,0.10); }}
    .topbar-kbd {{
      font-family: ui-monospace, SFMono-Regular, Consolas, Menlo, monospace;
      background: rgba(255,255,255,.08);
      padding: 1px 5px;
      border-radius: 4px;
      font-size: 12px;
    }}

    /* ── messages ────────────────────────────────────────────────────── */
    .messages {{
      flex: 1;
      overflow-y: auto;
      padding: 12px 0 24px;
    }}

    .msg-row {{
      display: flex;
      padding: 8px 24px;
      gap: 14px;
      max-width: 860px;
      margin: 0 auto;
      width: 100%;
    }}
    .msg-row.user {{ flex-direction: row-reverse; }}

    .avatar {{
      width: 32px; height: 32px; border-radius: 50%;
      flex-shrink: 0;
      display: flex; align-items: center; justify-content: center;
      font-size: 12px; font-weight: 700; margin-top: 2px;
    }}
    .avatar.bot  {{ background: var(--accent); color: #fff; }}
    .avatar.user {{ background: #5b5fc7; color: #fff; }}
    .avatar.sys  {{ background: #6b7280; color: #fff; font-size: 10px; }}

    .msg-content {{ flex: 1; min-width: 0; }}
    .msg-row.user .msg-content {{ text-align: right; }}

    .msg-bubble {{
      display: inline-block;
      max-width: 100%;
      text-align: left;
    }}
    .msg-row.user .msg-bubble {{
      background: var(--user-bg);
      border: 1px solid var(--border);
      border-radius: 18px 18px 6px 18px;
      padding: 10px 14px;
    }}
    .msg-row.bot .msg-bubble {{
      border-radius: 18px 18px 18px 6px;
    }}
    .msg-row.bot.script .msg-bubble {{ display: block; }}

    .msg-text {{ white-space: pre-wrap; word-wrap: break-word; }}
    .msg-meta {{ margin-top: 5px; font-size: 11px; color: var(--muted); }}

    /* code blocks */
    .code-wrap {{
      margin-top: 8px;
      border: 1px solid rgba(125,168,255,0.22);
      border-radius: 12px;
      background: #0d0d0d;
      overflow: hidden;
    }}
    .code-header {{
      display: flex; justify-content: flex-end;
      padding: 8px 10px 0;
    }}
    .copy-btn {{
      border: 1px solid rgba(148,163,184,0.2);
      background: rgba(255,255,255,0.04);
      color: var(--text);
      border-radius: 8px;
      padding: 4px 10px;
      font-size: 12px;
      cursor: pointer;
    }}
    .copy-btn:hover {{ background: rgba(255,255,255,0.08); }}
    .code-wrap pre {{
      margin: 0; padding: 10px 16px 14px;
      color: #c8d5f5;
      font-size: 13px; line-height: 1.55;
      overflow-x: auto;
      font-family: ui-monospace, SFMono-Regular, Consolas, Menlo, monospace;
    }}

    /* thinking indicator */
    .thinking-dots span {{
      display: inline-block;
      width: 6px; height: 6px;
      border-radius: 50%;
      background: var(--muted);
      margin: 0 2px;
      animation: bounce 1.2s infinite ease-in-out;
    }}
    .thinking-dots span:nth-child(1) {{ animation-delay: 0s; }}
    .thinking-dots span:nth-child(2) {{ animation-delay: .2s; }}
    .thinking-dots span:nth-child(3) {{ animation-delay: .4s; }}
    @keyframes bounce {{
      0%, 80%, 100% {{ transform: translateY(0); opacity: .5; }}
      40%            {{ transform: translateY(-5px); opacity: 1; }}
    }}

    /* ── composer ────────────────────────────────────────────────────── */
    .composer-wrap {{
      position: relative;
      flex-shrink: 0;
      padding: 0 20px 20px;
      max-width: 860px;
      margin: 0 auto;
      width: 100%;
    }}

    .cmd-popup {{
      position: absolute;
      bottom: calc(100% - 4px);
      left: 20px; right: 20px;
      background: var(--popup-bg);
      border: 1px solid var(--border);
      border-radius: 14px;
      overflow: hidden;
      box-shadow: var(--shadow-lg);
      display: none;
      z-index: 200;
      max-height: 320px;
      overflow-y: auto;
    }}
    .cmd-popup-header {{
      padding: 8px 16px 6px;
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: .09em;
      color: var(--muted);
      border-bottom: 1px solid var(--border);
    }}
    .cmd-item {{
      display: flex;
      align-items: center;
      gap: 14px;
      padding: 9px 16px;
      width: 100%;
      background: none;
      border: none;
      cursor: pointer;
      text-align: left;
      border-radius: 0;
      color: var(--text);
      font-size: 14px;
      transition: background .1s;
    }}
    .cmd-item:hover, .cmd-item.active {{
      background: var(--popup-hover);
    }}
    .cmd-item-name {{
      font-family: ui-monospace, SFMono-Regular, Consolas, Menlo, monospace;
      font-weight: 700;
      color: var(--accent);
      min-width: 130px;
      flex-shrink: 0;
    }}
    .cmd-item-desc {{ color: var(--muted); font-size: 13px; }}

    .composer-box {{
      display: flex;
      align-items: flex-end;
      gap: 10px;
      background: var(--input-bg);
      border: 1.5px solid var(--border);
      border-radius: 16px;
      padding: 10px 12px 10px 16px;
      transition: border-color .15s;
    }}
    .composer-box:focus-within {{
      border-color: var(--accent);
    }}
    .composer-box textarea {{
      flex: 1;
      background: none;
      border: none;
      color: var(--text);
      font: inherit;
      font-size: 15px;
      resize: none;
      min-height: 26px;
      max-height: 160px;
      outline: none;
      padding: 0;
      line-height: 1.55;
    }}
    .composer-box textarea::placeholder {{ color: var(--muted); }}
    .send-btn {{
      background: var(--accent);
      color: #fff;
      border: none;
      border-radius: 10px;
      width: 36px; height: 36px;
      display: flex; align-items: center; justify-content: center;
      cursor: pointer;
      flex-shrink: 0;
      transition: opacity .15s, transform .1s;
      font-size: 16px;
    }}
    .send-btn:hover {{ opacity: .85; transform: translateY(-1px); }}
    .send-btn:disabled {{ opacity: .4; cursor: not-allowed; transform: none; }}

    .composer-hint {{
      margin-top: 7px;
      font-size: 12px;
      color: var(--muted);
      text-align: center;
    }}

    /* ── mobile ──────────────────────────────────────────────────────── */
    @media (max-width: 700px) {{
      .sidebar {{ display: none; }}
      .msg-row {{ padding: 8px 14px; }}
      .composer-wrap {{ padding: 0 12px 14px; }}
    }}
  </style>
</head>
<body>
  <div class="app">

    <!-- ── sidebar ──────────────────────────────────────────── -->
    <aside class="sidebar">
      <div class="brand">
        <div class="brand-icon">🤖</div>
        <div>
          <div class="brand-title">Custom LLM</div>
          <div class="brand-sub">Local AI — no cloud</div>
        </div>
      </div>

      <div>
        <div class="sidebar-section-title">Model</div>
        <div class="stat-grid">
          <div class="stat-chip full">
            <div class="s-label">Backend</div>
            <div class="s-val">{backend_label}</div>
          </div>
          <div class="stat-chip">
            <div class="s-label">Params</div>
            <div class="s-val">{params_text}</div>
          </div>
          <div class="stat-chip">
            <div class="s-label">Vocab</div>
            <div class="s-val">{vocab_text}</div>
          </div>
          <div class="stat-chip full">
            <div class="s-label">Corpus docs</div>
            <div class="s-val">{corpus_text}</div>
          </div>
        </div>
      </div>

      <div>
        <div class="sidebar-section-title">Quick prompts</div>
        <div class="prompt-list" id="prompts">
          <button class="prompt-btn" data-prompt="hello">👋 Say hello</button>
          <button class="prompt-btn" data-prompt="what can you do">🤔 What can you do?</button>
          <button class="prompt-btn" data-prompt="generate python script to organize files by extension">🐍 Python file organizer</button>
          <button class="prompt-btn" data-prompt="what are ongoing policy challenges in usa">🇺🇸 USA policy challenges</button>
        </div>
      </div>

      <div class="sidebar-status" id="sidebarStatus">{version_text}</div>
    </aside>

    <!-- ── main ─────────────────────────────────────────────── -->
    <div class="main">

      <!-- topbar -->
      <div class="topbar">
        <span class="topbar-title">Chat — type <kbd class="topbar-kbd">/</kbd> for commands</span>
        <div class="topbar-actions">
          <button class="topbar-btn" id="retrainBtn">⟳ Retrain</button>
          <button class="topbar-btn danger" id="clearBtn">Clear chat</button>
        </div>
      </div>

      <!-- messages -->
      <div class="messages" id="messages"></div>

      <!-- composer -->
      <div class="composer-wrap">
        <div class="cmd-popup" id="cmdPopup">
          <div class="cmd-popup-header">Commands</div>
          <div id="cmdItems"></div>
        </div>
        <div class="composer-box">
          <textarea id="input" placeholder="Message Custom LLM — type / for commands" rows="1"></textarea>
          <button class="send-btn" id="sendBtn" title="Send (Enter)">&#9650;</button>
        </div>
        <div class="composer-hint" id="hint">Enter to send &nbsp;·&nbsp; Shift+Enter for newline &nbsp;·&nbsp; / for commands</div>
      </div>
    </div>
  </div>

  <script>
    /* ── data ─────────────────────────────────────────────────────── */
    const COMMANDS = {cmds_json};

    /* ── refs ─────────────────────────────────────────────────────── */
    const messagesEl = document.getElementById('messages');
    const inputEl    = document.getElementById('input');
    const sendBtn    = document.getElementById('sendBtn');
    const clearBtn   = document.getElementById('clearBtn');
    const retrainBtn = document.getElementById('retrainBtn');
    const cmdPopup   = document.getElementById('cmdPopup');
    const cmdItems   = document.getElementById('cmdItems');
    const hintEl     = document.getElementById('hint');
    const sidebarStatus = document.getElementById('sidebarStatus');
    const promptList = document.getElementById('prompts');

    /* ── state ────────────────────────────────────────────────────── */
    let popupSel = 0;
    let isWaiting = false;

    /* ── auto-resize textarea ─────────────────────────────────────── */
    function autoResize() {{
      inputEl.style.height = 'auto';
      inputEl.style.height = Math.min(inputEl.scrollHeight, 160) + 'px';
    }}
    inputEl.addEventListener('input', autoResize);

    /* ── status helper ────────────────────────────────────────────── */
    function setHint(txt) {{
      hintEl.textContent = txt;
      sidebarStatus.textContent = txt;
    }}

    /* ── scroll ───────────────────────────────────────────────────── */
    function scrollBottom() {{
      messagesEl.scrollTop = messagesEl.scrollHeight;
    }}

    /* ── message rendering ────────────────────────────────────────── */
    function makeCopyBtn(code) {{
      const btn = document.createElement('button');
      btn.className = 'copy-btn';
      btn.textContent = 'Copy';
      btn.addEventListener('click', async () => {{
        try {{
          await navigator.clipboard.writeText(code);
          btn.textContent = 'Copied!';
          setTimeout(() => {{ btn.textContent = 'Copy'; }}, 1400);
        }} catch (_) {{
          btn.textContent = 'Failed';
          setTimeout(() => {{ btn.textContent = 'Copy'; }}, 1400);
        }}
      }});
      return btn;
    }}

    function makeCodeBlock(code) {{
      const wrap = document.createElement('div');
      wrap.className = 'code-wrap';
      const hdr = document.createElement('div');
      hdr.className = 'code-header';
      hdr.appendChild(makeCopyBtn(code));
      const pre = document.createElement('pre');
      pre.textContent = code;
      wrap.appendChild(hdr);
      wrap.appendChild(pre);
      return wrap;
    }}

    function renderMessageContent(text, isScript) {{
      const frag = document.createDocumentFragment();

      if (isScript) {{
        const usageIdx = text.indexOf('\\nUsage:');
        let usage = '';
        let body = text.trim();
        if (usageIdx >= 0) {{
          usage = body.slice(usageIdx + 1).trim();
          body = body.slice(0, usageIdx).trimEnd();
        }}
        const sep = '\\n\\n';
        const sepIdx = body.indexOf(sep);
        let intro = '', code = body;
        if (sepIdx >= 0) {{
          intro = body.slice(0, sepIdx).trim();
          code  = body.slice(sepIdx + sep.length).trim();
        }}
        if (intro) {{
          const p = document.createElement('div');
          p.className = 'msg-text';
          p.style.marginBottom = '10px';
          p.textContent = intro;
          frag.appendChild(p);
        }}
        if (code) frag.appendChild(makeCodeBlock(code));
        if (usage) {{
          const u = document.createElement('div');
          u.className = 'msg-text';
          u.style.marginTop = '10px';
          u.style.color = 'var(--muted)';
          u.textContent = usage;
          frag.appendChild(u);
        }}
      }} else {{
        const parts = text.split(/(```[\\s\\S]*?```)/g);
        for (const part of parts) {{
          if (part.startsWith('```') && part.endsWith('```')) {{
            const inner = part.slice(3, -3).replace(/^[^\\n]*\\n/, '').trimEnd();
            frag.appendChild(makeCodeBlock(inner || part.slice(3, -3)));
          }} else if (part) {{
            const d = document.createElement('div');
            d.className = 'msg-text';
            d.textContent = part;
            frag.appendChild(d);
          }}
        }}
      }}
      return frag;
    }}

    function addMessage(text, role, meta, source) {{
      const isScript = role === 'bot' && source === 'python-script-generator';
      const row = document.createElement('div');
      row.className = 'msg-row ' + role + (isScript ? ' script' : '');

      const av = document.createElement('div');
      av.className = 'avatar ' + (role === 'user' ? 'user' : role === 'system' ? 'sys' : 'bot');
      av.textContent = role === 'user' ? 'You' : role === 'system' ? 'SYS' : 'AI';

      const content = document.createElement('div');
      content.className = 'msg-content';

      const bubble = document.createElement('div');
      bubble.className = 'msg-bubble';
      bubble.appendChild(renderMessageContent(text, isScript));

      if (meta) {{
        const m = document.createElement('div');
        m.className = 'msg-meta';
        m.textContent = meta;
        bubble.appendChild(m);
      }}

      content.appendChild(bubble);
      row.appendChild(av);
      row.appendChild(content);
      messagesEl.appendChild(row);
      scrollBottom();
      return row;
    }}

    function addThinking() {{
      const row = document.createElement('div');
      row.className = 'msg-row bot';
      row.id = '__thinking__';

      const av = document.createElement('div');
      av.className = 'avatar bot';
      av.textContent = 'AI';

      const content = document.createElement('div');
      content.className = 'msg-content';
      const bubble = document.createElement('div');
      bubble.className = 'msg-bubble';
      const dots = document.createElement('div');
      dots.className = 'thinking-dots';
      dots.innerHTML = '<span></span><span></span><span></span>';
      bubble.appendChild(dots);
      content.appendChild(bubble);
      row.appendChild(av);
      row.appendChild(content);
      messagesEl.appendChild(row);
      scrollBottom();
      return row;
    }}

    function removeThinking() {{
      const el = document.getElementById('__thinking__');
      if (el) el.remove();
    }}

    /* ── command popup ────────────────────────────────────────────── */
    function filteredCmds(prefix) {{
      const p = prefix.toLowerCase().replace(/-/g, '_');
      return COMMANDS.filter(c => c.name.startsWith(p));
    }}

    function renderCmdPopup(list) {{
      cmdItems.innerHTML = '';
      list.forEach((cmd, i) => {{
        const btn = document.createElement('button');
        btn.className = 'cmd-item' + (i === popupSel ? ' active' : '');
        btn.setAttribute('data-idx', String(i));

        const nm = document.createElement('span');
        nm.className = 'cmd-item-name';
        nm.textContent = '/' + cmd.name;

        const ds = document.createElement('span');
        ds.className = 'cmd-item-desc';
        ds.textContent = cmd.desc;

        btn.appendChild(nm);
        btn.appendChild(ds);
        btn.addEventListener('mousedown', (e) => {{
          e.preventDefault();
          applyCommand(cmd);
        }});
        cmdItems.appendChild(btn);
      }});
    }}

    function showPopup(prefix) {{
      const list = filteredCmds(prefix);
      if (!list.length) {{ hidePopup(); return; }}
      popupSel = Math.min(popupSel, list.length - 1);
      renderCmdPopup(list);
      cmdPopup.style.display = 'block';
    }}

    function hidePopup() {{
      cmdPopup.style.display = 'none';
      popupSel = 0;
    }}

    function applyCommand(cmd) {{
      const needsArg = ['search'].includes(cmd.name);
      inputEl.value = '/' + cmd.name + (needsArg ? ' ' : '');
      hidePopup();
      autoResize();
      inputEl.focus();
      if (!needsArg) submitInput();
    }}

    function popupVisible() {{
      return cmdPopup.style.display === 'block';
    }}

    /* ── input event ──────────────────────────────────────────────── */
    inputEl.addEventListener('input', () => {{
      const val = inputEl.value;
      if (val.startsWith('/')) {{
        popupSel = 0;
        showPopup(val.slice(1));
      }} else {{
        hidePopup();
      }}
    }});

    /* ── keyboard ─────────────────────────────────────────────────── */
    inputEl.addEventListener('keydown', (e) => {{
      if (popupVisible()) {{
        const list = filteredCmds(inputEl.value.slice(1));
        if (e.key === 'ArrowDown') {{
          e.preventDefault();
          popupSel = Math.min(popupSel + 1, list.length - 1);
          renderCmdPopup(list);
          return;
        }}
        if (e.key === 'ArrowUp') {{
          e.preventDefault();
          popupSel = Math.max(0, popupSel - 1);
          renderCmdPopup(list);
          return;
        }}
        if (e.key === 'Tab' || e.key === 'Enter') {{
          e.preventDefault();
          if (list[popupSel]) applyCommand(list[popupSel]);
          return;
        }}
        if (e.key === 'Escape') {{
          e.preventDefault();
          hidePopup();
          return;
        }}
      }}
      if (e.key === 'Enter' && !e.shiftKey) {{
        e.preventDefault();
        submitInput();
      }}
    }});

    /* ── fetch helper ─────────────────────────────────────────────── */
    async function fetchJson(url, opts) {{
      const r = await fetch(url, opts);
      const d = await r.json();
      if (!r.ok) throw new Error(d.error || d.message || 'Request failed');
      return d;
    }}

    /* ── command handlers ─────────────────────────────────────────── */
    async function cmdHelp() {{
      const lines = COMMANDS.map(c => ('/' + c.name).padEnd(16) + c.desc);
      addMessage(lines.join('\\n'), 'system', 'Command list', 'plain');
    }}

    async function cmdVersion() {{
      try {{
        const d = await fetchJson('/api/meta');
        addMessage(
          'Web UI:  ' + d.web_ui_version + '\\nCLI UI:  ' + d.cli_version + '\\nLLM:     ' + d.llm_version,
          'system', 'Versions', 'plain'
        );
      }} catch (err) {{
        addMessage('Error: ' + err.message, 'system', 'error', 'plain');
      }}
    }}

    async function cmdModelInfo() {{
      try {{
        const d = await fetchJson('/api/model_info');
        const lines = [
          'Backend:   ' + (d.backend || '?').toUpperCase(),
          'Model:     ' + (d.model_name || '?'),
          'Device:    ' + (d.device || 'cpu'),
        ];
        if (d.num_parameters) lines.push('Params:    ' + Number(d.num_parameters).toLocaleString());
        if (d.vocab_size)     lines.push('Vocab:     ' + Number(d.vocab_size).toLocaleString());
        if (d.d_model)        lines.push('d_model:   ' + d.d_model);
        if (d.n_layers)       lines.push('n_layers:  ' + d.n_layers);
        if (d.n_heads)        lines.push('n_heads:   ' + d.n_heads);
        if (d.context_length) lines.push('ctx_len:   ' + d.context_length);
        if (d.corpus_size)    lines.push('Corpus:    ' + Number(d.corpus_size).toLocaleString() + ' docs');
        if (d.epochs)         lines.push('Epochs:    ' + d.epochs);
        if (d.status)         lines.push('', d.status);
        lines.push('', 'Use /retrain to rebuild model with your conversation history.');
        addMessage(lines.join('\\n'), 'system', 'Model info', 'plain');
      }} catch (err) {{
        addMessage('Error: ' + err.message, 'system', 'error', 'plain');
      }}
    }}

    async function cmdGpuStatus() {{
      try {{
        const d = await fetchJson('/api/gpu_status');
        addMessage(d.status, 'system', 'GPU status', 'plain');
      }} catch (err) {{
        addMessage('Error: ' + err.message, 'system', 'error', 'plain');
      }}
    }}

    async function cmdRefresh() {{
      setHint('Refreshing session...');
      try {{
        await fetchJson('/api/refresh', {{ method: 'POST' }});
        window.location.reload();
      }} catch (err) {{
        addMessage('Error: ' + err.message, 'system', 'error', 'plain');
        setHint('Refresh failed.');
      }}
    }}

    async function cmdRetrain() {{
      setHint('Retraining... this may take a few minutes.');
      retrainBtn.disabled = true;
      const tRow = addMessage(
        'Starting retrain — reading all knowledge files and your conversation history. Please wait...',
        'system', null, 'plain'
      );
      try {{
        const d = await fetchJson('/api/retrain', {{ method: 'POST' }});
        addMessage(d.message || 'Retrain complete.', 'system', 'Retrain result', 'plain');
        setHint('Retrain complete. Reload to refresh model stats.');
      }} catch (err) {{
        addMessage('Retrain failed: ' + err.message, 'system', 'error', 'plain');
        setHint('Retrain failed.');
      }} finally {{
        retrainBtn.disabled = false;
      }}
    }}

    /* ── chat ─────────────────────────────────────────────────────── */
    async function sendChat(msg, forceWeb) {{
      addMessage(msg, 'user');
      inputEl.value = '';
      autoResize();
      setHint('Thinking...');
      sendBtn.disabled = true;

      const thinking = addThinking();
      try {{
        const d = await fetchJson('/api/chat', {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/json' }},
          body: JSON.stringify({{ message: msg, force_web: !!forceWeb }})
        }});
        removeThinking();
        const meta = 'source=' + d.source + ' | confidence=' + (d.confidence != null ? d.confidence : 'n/a');
        addMessage(d.answer, 'bot', meta, d.source);
        setHint('Ready.');
      }} catch (err) {{
        removeThinking();
        addMessage('Error: ' + err.message, 'system', 'request failed', 'plain');
        setHint('Request failed.');
      }} finally {{
        sendBtn.disabled = false;
        inputEl.focus();
      }}
    }}

    /* ── submit ───────────────────────────────────────────────────── */
    async function submitInput() {{
      if (isWaiting) return;
      const val = inputEl.value.trim();
      if (!val) return;

      hidePopup();

      if (val.startsWith('/')) {{
        inputEl.value = '';
        autoResize();
        const rest  = val.slice(1).trim();
        const parts = rest.split(/\\s+/, 2);
        const cmd   = parts[0].toLowerCase().replace(/-/g, '_');
        const arg   = parts.length > 1 ? parts[1] : '';

        if (!cmd)                       {{ await cmdHelp(); return; }}
        if (cmd === 'help')             {{ await cmdHelp(); return; }}
        if (cmd === 'version')          {{ await cmdVersion(); return; }}
        if (cmd === 'model_info')       {{ await cmdModelInfo(); return; }}
        if (cmd === 'gpu' || cmd === 'gpu_status') {{ await cmdGpuStatus(); return; }}
        if (cmd === 'refresh')          {{ await cmdRefresh(); return; }}
        if (cmd === 'retrain')          {{ await cmdRetrain(); return; }}
        if (cmd === 'clear')            {{ clearChat(); return; }}
        if (cmd === 'exit')             {{ addMessage('Close this browser tab to exit.', 'system', null, 'plain'); return; }}
        if (cmd === 'search') {{
          if (!arg) {{ addMessage('Usage: /search <query>', 'system', null, 'plain'); return; }}
          await sendChat(arg, true);
          return;
        }}
        addMessage('Unknown command. Type /help for the list.', 'system', null, 'plain');
        return;
      }}

      if (val.toLowerCase().startsWith('search:')) {{
        const q = val.slice(7).trim();
        if (!q) {{ addMessage('Provide a query after "search:"', 'system', null, 'plain'); return; }}
        await sendChat(q, true);
        return;
      }}

      await sendChat(val, false);
    }}

    /* ── misc UI ──────────────────────────────────────────────────── */
    function clearChat() {{
      messagesEl.innerHTML = '';
      addMessage('Chat cleared. Ask anything or type / for commands.', 'system', null, 'plain');
      setHint('Ready.');
    }}

    /* ── wire up buttons ──────────────────────────────────────────── */
    sendBtn.addEventListener('click', submitInput);
    clearBtn.addEventListener('click', clearChat);
    retrainBtn.addEventListener('click', cmdRetrain);

    promptList.addEventListener('click', (e) => {{
      const btn = e.target.closest('button[data-prompt]');
      if (!btn) return;
      inputEl.value = btn.dataset.prompt;
      autoResize();
      inputEl.focus();
      setHint('Prompt loaded — press Enter to send.');
    }});

    document.addEventListener('click', (e) => {{
      if (!cmdPopup.contains(e.target) && e.target !== inputEl) hidePopup();
    }});

    /* ── init ─────────────────────────────────────────────────────── */
    addMessage(
      'Hello! I am your local Custom LLM assistant.\\n\\nAsk me anything, or type / to see available commands. Your conversations are remembered and used to improve the model when you run /retrain.',
      'bot', null, 'plain'
    );
    setHint('Enter to send  ·  Shift+Enter for newline  ·  / for commands');
    inputEl.focus();
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
