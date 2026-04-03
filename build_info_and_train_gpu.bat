@echo off
setlocal

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo [build-train-gpu] Environment missing. Running setup first...
  call setup.bat
)

set PYTHON=%cd%\.venv\Scripts\python.exe

call :ensure_sentence_transformers
if errorlevel 1 exit /b 1

call :check_torch_ready
if not errorlevel 1 goto torch_ready

echo [build-train-gpu] Building training data from knowledge files...
set PYTHONPATH=%cd%\src
call "%PYTHON%" -m custom_llm.build_training_data
if errorlevel 1 (
  echo [build-train-gpu] Failed while building training data.
  exit /b 1
)

call :install_torch_cuda
if errorlevel 1 exit /b 1

:torch_ready
echo [build-train-gpu] Torch/CUDA already compatible. Skipping reinstall.

echo [build-train-gpu] Verifying torch version and CUDA runtime...
call "%PYTHON%" -c "import torch,sys; v=torch.__version__; c=(torch.version.cuda or ''); ok=v.startswith('2.9') and c.startswith('13.'); print(f'[build-train-gpu] torch={v} cuda={c or 'unknown'} cuda_available={torch.cuda.is_available()}'); sys.exit(0 if ok and torch.cuda.is_available() else 1)"
if errorlevel 1 (
  echo [build-train-gpu] Torch/CUDA check failed. Need torch 2.9.x with CUDA 13.x and cuda_available=True.
  exit /b 1
)

echo [build-train-gpu] Training with transformer backend (GPU preferred)...
call "%PYTHON%" -m custom_llm.trainer --backend transformer
if errorlevel 1 (
  echo [build-train-gpu] Transformer backend unavailable; falling back to auto backend.
  call "%PYTHON%" -m custom_llm.trainer --backend auto
)

echo [build-train-gpu] Done.
endlocal
exit /b 0

:ensure_sentence_transformers
echo [build-train-gpu] Checking sentence-transformers dependency...
call "%PYTHON%" -c "import sentence_transformers" >nul 2>nul
if not errorlevel 1 exit /b 0
echo [build-train-gpu] Installing sentence-transformers...
call "%PYTHON%" -m pip install -r requirements-gpu.txt
if errorlevel 1 (
  echo [build-train-gpu] Failed to install sentence-transformers.
  exit /b 1
)
exit /b 0

:check_torch_ready
call "%PYTHON%" -c "import torch,sys; v=torch.__version__; c=(torch.version.cuda or ''); ok=v.startswith('2.9') and c.startswith('13.') and torch.cuda.is_available(); sys.exit(0 if ok else 1)"
exit /b %errorlevel%

:install_torch_cuda
set CUDA_TAG=cu132
call :try_install
if not errorlevel 1 exit /b 0

set CUDA_TAG=cu131
call :try_install
if not errorlevel 1 exit /b 0

set CUDA_TAG=cu130
call :try_install
if not errorlevel 1 exit /b 0

echo [build-train-gpu] Could not install torch 2.9 CUDA wheels from cu132/cu131/cu130 channels.
exit /b 1

:try_install
set PYTORCH_CUDA_WHL_INDEX=https://download.pytorch.org/whl/%CUDA_TAG%
echo [build-train-gpu] Trying PyTorch 2.9 on %CUDA_TAG% channel...
call "%PYTHON%" -m pip install --upgrade --index-url %PYTORCH_CUDA_WHL_INDEX% "torch>=2.9,<2.10"
if errorlevel 1 (
  echo [build-train-gpu] %CUDA_TAG% wheel channel did not match this environment.
  exit /b 1
)
echo [build-train-gpu] Installed torch from %CUDA_TAG% channel.
exit /b 0
