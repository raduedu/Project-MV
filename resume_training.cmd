@echo off
rem Project MV: RESUME the training plan exactly where it stopped (then the rest of the plan).
cd /d "%~dp0"
".venv\Scripts\python.exe" scripts\train_plan.py resume
echo.
pause
