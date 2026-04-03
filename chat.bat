@echo off
setlocal

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo [chat] Environment missing. Running setup first...
  call setup.bat
)

if not exist "artifacts\vectorizer.joblib" (
  echo [chat] No trained model artifacts found. Running training first...
  call train.bat
)

set PYTHONPATH=%cd%\src
call .venv\Scripts\python.exe -m custom_llm.cli

endlocal
