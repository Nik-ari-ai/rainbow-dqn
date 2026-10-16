@echo off
cd /d "%~dp0"
if not exist results mkdir results
"%USERPROFILE%\rl_venv\Scripts\python.exe" -u record_best.py --model runs\rainbow_pixels\model.pth --candidates 20 --out results\rainbow_best_game.mp4 > logs\best_video.log 2>&1
echo finished>> logs\best_video.log
