@echo off
rem Project MV: show what is training, what is done, paused or waiting.
cd /d "%~dp0"
".venv\Scripts\python.exe" scripts\train_plan.py status
echo.
pause
