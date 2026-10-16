@echo off
rem Full pipeline: Rainbow (2M steps) -> evaluation vs random -> demo video -> vanilla DQN baseline -> final comparison.
rem Progress: logs\run_all.log (summary) and logs\<run>.log (details). Close this window to stop; rerun to resume.
cd /d "%~dp0"
if not exist logs mkdir logs
set "VPY=%USERPROFILE%\rl_venv\Scripts\python.exe"
call :main >> logs\run_all.log 2>&1
exit /b

:run
rem %1 = run id, rest = extra args
set "RID=%~1"
set "RESUME="
if exist runs\%RID%\checkpoint.pt set "RESUME=--resume"
echo [%date% %time%] start %RID% %RESUME%
"%VPY%" -u train.py --id %* %RESUME% > logs\%RID%.log 2>&1
echo [%date% %time%] end %RID% (exit %errorlevel%)
exit /b

:main
echo ===== run_all started %date% %time%
call :run rainbow_pixels --steps 2000000
echo [%date% %time%] evaluate Rainbow vs random
"%VPY%" -u evaluate.py --models runs\rainbow_pixels\model.pth --names Rainbow --episodes 30 --device cuda --out results > logs\evaluate.log 2>&1
echo [%date% %time%] demo video
"%VPY%" -u demo.py --model runs\rainbow_pixels\model.pth --compare-random --out results\demo_vs_random.mp4 > logs\demo.log 2>&1
call :run dqn --dqn --steps 2000000
echo [%date% %time%] final comparison
"%VPY%" -u evaluate.py --models runs\rainbow_pixels\model.pth runs\dqn\model.pth --names Rainbow DQN --episodes 30 --device cuda --out results > logs\evaluate_final.log 2>&1
echo ===== run_all finished %date% %time%
exit /b
