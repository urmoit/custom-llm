from __future__ import annotations

import json
import threading
import time
import os
import importlib
import sys
import shutil
import textwrap
from typing import Callable, Dict

from .assistant import SmartAssistant
from .config import MODEL_META_FILE


class _Style:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    MAGENTA = "\033[35m"


def _supports_color() -> bool:
    return sys.stdout.isatty()


def _paint(text: str, color: str = "", bold: bool = False, dim: bool = False) -> str:
    if not _supports_color():
        return text

    parts = []
    if bold:
        parts.append(_Style.BOLD)
    if dim:
        parts.append(_Style.DIM)
    if color:
        parts.append(color)
    return "".join(parts) + text + _Style.RESET


def _box(title: str, lines: list[str], color: str = "") -> str:
    width = min(max(shutil.get_terminal_size((80, 24)).columns - 2, 60), 100)
    inner = width - 4

    rendered_lines: list[str] = []
    for line in lines:
        if not line:
            rendered_lines.append("")
            continue
        rendered_lines.extend(textwrap.wrap(line, width=inner) or [""])

    top = "+" + "-" * (width - 2) + "+"
    title_line = f"| {_paint(title[:inner], color=color, bold=True).ljust(inner)} |"

    body = []
    for line in rendered_lines:
        body.append(f"| {line.ljust(inner)} |")

    return "\n".join([top, title_line] + body + [top])


def _current_versions() -> tuple[str, str]:
    version_module = importlib.import_module("custom_llm.version")
    version_module = importlib.reload(version_module)
    cli_version = str(getattr(version_module, "CLI_UI_VERSION", version_module.VERSION))
    llm_version = str(getattr(version_module, "LLM_MODEL_VERSION", cli_version))
    return cli_version, llm_version


def _welcome_text() -> str:
    cli_version, llm_version = _current_versions()
    header = _box(
        f"Custom LLM CLI  v{cli_version}",
        [
            "Ask anything. For conversions, try: how many cm are in 2 km",
            f"CLI UI version: v{cli_version} | LLM model version: v{llm_version}",
            "",
            "Commands:",
            "/help      Show commands",
            "/version   Show CLI and model versions",
            "/gpu_status Show CUDA and training backend status",
            "           aliases: /gpu-status, /gpu",
            "/search    Force web search (example: /search latest Python)",
            "/refresh   Refresh chat session",
            "/clear     Clear screen",
            "/exit      Quit",
            "Tip: type / to list commands, or /re to filter",
        ],
        color=_Style.CYAN,
    )
    return header


def _print_bot_block(body: str, title: str = "") -> None:
    color = _Style.GREEN
    heading = "Bot"

    if title:
        heading = f"Bot | {title}"
    if "error" in title.lower():
        color = _Style.RED
    elif "system" in title.lower():
        color = _Style.YELLOW
    elif "web-search" in title.lower():
        color = _Style.MAGENTA

    print()
    print(_box(heading, body.splitlines() or [""], color=color))
    print()


def _run_with_spinner(task: Callable[[], object], label: str = "Thinking"):
    done = False
    result_holder = {"value": None, "error": None}

    def runner() -> None:
        try:
            result_holder["value"] = task()
        except Exception as exc:
            result_holder["error"] = exc
        finally:
            nonlocal done
            done = True

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()

    frames = [".", "..", "...", "...."]
    idx = 0
    while not done:
        frame = frames[idx % len(frames)]
        text = _paint(label, color=_Style.CYAN, bold=True)
        print(f"\r{text} {frame}", end="", flush=True)
        idx += 1
        time.sleep(0.12)

    print("\r" + " " * 40 + "\r", end="", flush=True)

    if result_holder["error"] is not None:
        raise result_holder["error"]
    return result_holder["value"]


def _clear_screen_if_interactive() -> None:
    if sys.stdin.isatty() and sys.stdout.isatty():
        os.system("cls")


def _print_user_prompt() -> str:
    return _paint("You", color=_Style.CYAN, bold=True) + "> "


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

    if MODEL_META_FILE.exists():
        try:
            meta = json.loads(MODEL_META_FILE.read_text(encoding="utf-8"))
            lines.append("")
            lines.append("Last trained model metadata:")
            lines.append(f"- backend: {meta.get('backend', 'unknown')}")
            lines.append(f"- model: {meta.get('model_name', 'unknown')}")
            lines.append(f"- device: {meta.get('device', 'unknown')}")
            note = str(meta.get("note", "")).strip()
            if note:
                lines.append(f"- note: {note}")
        except Exception as exc:
            lines.append(f"Could not read model metadata: {exc}")
    else:
        lines.append("")
        lines.append("No model metadata found yet. Train first.")

    return "\n".join(lines)


def _command_specs() -> Dict[str, str]:
    return {
        "help": "Show commands",
        "version": "Show CLI and model versions",
        "gpu_status": "Show CUDA and training backend status",
        "search": "Force web search",
        "refresh": "Refresh chat session",
        "clear": "Clear screen",
        "exit": "Quit",
    }


def _show_command_suggestions(prefix: str = "") -> None:
    specs = _command_specs()
    prefix = prefix.strip().lower().replace("-", "_")

    lines = []
    for name, desc in specs.items():
        if prefix and not name.startswith(prefix):
            continue
        marker = f"/{name}"
        if prefix:
            marker = f"/{name[:len(prefix)]}[{name[len(prefix):]}]"
        lines.append(f"{marker}  {desc}")

    if not lines:
        _print_bot_block("No command matches that prefix.", title="system")
        return

    lines.append("")
    lines.append("Type a full command and press Enter.")
    _print_bot_block("\n".join(lines), title="system")


def run_cli() -> None:
    bot = SmartAssistant()

    print(_welcome_text())

    if not bot.documents:
        print("[System] No trained model found. Run build_info_and_train.bat first.\n")

    while True:
        try:
            user_input = input(_print_user_prompt()).strip()
        except (EOFError, KeyboardInterrupt):
            print("\n" + _paint("Exiting.", color=_Style.YELLOW))
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            cmdline = user_input[1:].strip()
            if not cmdline:
                _show_command_suggestions("")
                continue

            parts = cmdline.split(" ", 1)
            cmd_name = parts[0].lower().replace("-", "_")
            cmd_arg = parts[1].strip() if len(parts) > 1 else ""

            if cmd_name in {"quit", "exit"}:
                print(_paint("Bye.", color=_Style.YELLOW))
                break

            if cmd_name == "help":
                _show_command_suggestions("")
                continue

            if cmd_name in {"gpu", "gpu_status"}:
                _print_bot_block(_gpu_status_text(), title="system")
                continue

            if cmd_name == "version":
                cli_version, llm_version = _current_versions()
                _print_bot_block(
                    f"CLI UI version: v{cli_version}\nLLM model version: v{llm_version}",
                    title="system",
                )
                continue

            if cmd_name == "clear":
                _clear_screen_if_interactive()
                print(_welcome_text())
                continue

            if cmd_name == "refresh":
                bot = SmartAssistant()
                _clear_screen_if_interactive()
                print(_welcome_text())
                _print_bot_block("Session refreshed.", title="system")
                continue

            if cmd_name == "search":
                if not cmd_arg:
                    _print_bot_block("Usage: /search <query>", title="error")
                    continue

                try:
                    result = _run_with_spinner(
                        lambda: bot.ask(cmd_arg, force_web=True),
                        label="Thinking",
                    )
                    confidence_text = (
                        f"{result.confidence:.2f}" if result.confidence is not None else "n/a"
                    )
                    body = f"Thought: {result.thinking}\n\n{result.answer}"
                    header = f"source={result.source} | confidence={confidence_text}"
                    _print_bot_block(body, title=header)
                except Exception as exc:
                    _print_bot_block(f"I hit an error: {exc}", title="error")
                continue

            _show_command_suggestions(cmd_name)
            continue

        force_web = False
        query = user_input
        if user_input.lower().startswith("search:"):
            force_web = True
            query = user_input.split(":", 1)[1].strip()
            if not query:
                _print_bot_block("Please provide a query after 'search:'.", title="error")
                continue

        try:
            result = _run_with_spinner(
                lambda: bot.ask(query, force_web=force_web),
                label="Thinking",
            )

            confidence_text = (
                f"{result.confidence:.2f}" if result.confidence is not None else "n/a"
            )
            body = f"Thought: {result.thinking}\n\n{result.answer}"
            if result.source == "web-search":
                header = f"source={result.source} | confidence={confidence_text}"
                _print_bot_block(body, title=header)
            else:
                _print_bot_block(body)
        except Exception as exc:
            _print_bot_block(f"I hit an error: {exc}", title="error")


if __name__ == "__main__":
    run_cli()
