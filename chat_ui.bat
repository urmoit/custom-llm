@echo off
setlocal
pushd "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo [chat-ui] Python environment not found. Run setup.bat first.
  popd
  exit /b 1
)

set "PYTHON=.venv\Scripts\python.exe"
set "HOST=127.0.0.1"
set "PORT=8787"
set "URL=http://%HOST%:%PORT%"
set "PYTHONPATH=%cd%\src"

echo [chat-ui] Starting web UI at %URL%
start "Custom LLM Web UI" /b "%PYTHON%" -m custom_llm.web_ui --host %HOST% --port %PORT%
timeout /t 2 /nobreak >nul
start "" "%URL%"

echo [chat-ui] Browser opened.
popd
endlocal
