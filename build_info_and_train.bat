@echo off
setlocal

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo [build-train] Environment missing. Running setup first...
  call setup.bat
)

echo [build-train] Building training data from knowledge files...
set PYTHONPATH=%cd%\src
call .venv\Scripts\python.exe -m custom_llm.build_training_data
if errorlevel 1 (
  echo [build-train] Failed while building training data.
  exit /b 1
)

echo [build-train] Training model from generated dataset...
call .venv\Scripts\python.exe -m custom_llm.trainer
if errorlevel 1 (
  echo [build-train] Failed while training model.
  exit /b 1
)

echo [build-train] Done.
endlocal
