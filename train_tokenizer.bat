@echo off
setlocal

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo [train-tokenizer] Environment missing. Running setup first...
  call setup.bat
)

echo [train-tokenizer] Building training data from knowledge and memory...
set PYTHONPATH=%cd%\src
call .venv\Scripts\python.exe -m custom_llm.build_training_data
if errorlevel 1 (
  echo [train-tokenizer] Failed while building training data.
  exit /b 1
)

echo [train-tokenizer] Training with custom tokenizer + transformer LLM backend...
set PYTHONPATH=%cd%\src
call .venv\Scripts\python.exe -m custom_llm.trainer --backend custom
if errorlevel 1 (
  echo [train-tokenizer] Training failed.
  exit /b 1
)

echo [train-tokenizer] Finished.
endlocal
