@echo off
rem Project MV: PAUSE all training (saves every driver, starts nothing new).
rem Safe to close the laptop once it says so. Continue later with resume_training.cmd
cd /d "%~dp0"
".venv\Scripts\python.exe" scripts\train_plan.py pause
echo.
pause
