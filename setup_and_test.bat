@echo off
rem Creates a Python venv with CUDA PyTorch (RTX 50xx needs cu128), runs tests and a 2-minute speed test.
rem Output: logs\setup.log, then logs\setup_status.txt = OK or FAIL
cd /d "%~dp0"
if not exist logs mkdir logs
echo running > logs\setup_status.txt
call :main > logs\setup.log 2>&1
if errorlevel 1 (echo FAIL > logs\setup_status.txt) else (echo OK > logs\setup_status.txt)
exit /b

:main
echo === start %date% %time%
nvidia-smi
where python
py -0p
set "PY="
py -3.12 -c "print(1)" >nul 2>&1 && set "PY=py -3.12"
if not defined PY py -3.11 -c "print(1)" >nul 2>&1 && set "PY=py -3.11"
if not defined PY py -3.13 -c "print(1)" >nul 2>&1 && set "PY=py -3.13"
if not defined PY python -c "print(1)" >nul 2>&1 && set "PY=python"
if not defined PY (echo NO PYTHON FOUND & exit /b 1)
echo using %PY%
set "VENV=%USERPROFILE%\rl_venv"
if not exist "%VENV%\Scripts\python.exe" %PY% -m venv "%VENV%" || exit /b 1
set "VPY=%VENV%\Scripts\python.exe"
"%VPY%" --version
"%VPY%" -m pip install --upgrade pip -q
"%VPY%" -m pip install torch --index-url https://download.pytorch.org/whl/cu128 || exit /b 1
"%VPY%" -m pip install numpy gymnasium pandas matplotlib scipy pillow imageio imageio-ffmpeg pygame tabulate || exit /b 1
"%VPY%" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
echo === tests
"%VPY%" -B tests\test_all.py || exit /b 1
echo === speed test (about 2 minutes)
"%VPY%" -u train.py --id speedtest --steps 200000 --learn-start 2000 --eval-interval 1000000 --log-interval 2000 --checkpoint-interval 0 --eval-episodes 3 --time-limit 0.035 || exit /b 1
echo === done %date% %time%
exit /b 0
