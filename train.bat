@echo off
setlocal

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo [train] Environment missing. Running setup first...
  call setup.bat
)

echo [train] Building training data from knowledge and memory...
set PYTHONPATH=%cd%\src
call .venv\Scripts\python.exe -m custom_llm.build_training_data
if errorlevel 1 (
  echo [train] Failed while building training data.
  exit /b 1
)

echo [train] Training local knowledge model...
set PYTHONPATH=%cd%\src
call .venv\Scripts\python.exe -m custom_llm.trainer

echo [train] Finished.
endlocal
