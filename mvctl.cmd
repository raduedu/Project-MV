@echo off
rem Project MV control.  From the project folder:   mvctl <command>     (in PowerShell: .\mvctl <command>)
rem
rem   mvctl status     what is training, done, paused or waiting (with progress bars)
rem   mvctl watch      the same, refreshed every 5 s - Ctrl+C to leave (training is not affected)
rem   mvctl pause      save every running driver and stop - safe to close the laptop afterwards
rem   mvctl resume     continue exactly where it stopped, then the rest of the plan
rem   mvctl ui         start the web interface (http://127.0.0.1:8765) and open it in the browser
rem   mvctl ui-stop    stop the web interface (training is not affected)
rem   mvctl remote     show the web interface on your Tailscale network (your devices only, HTTPS)
rem   mvctl remote-off stop showing it on Tailscale (the local web interface keeps running)
rem   mvctl tests      run the automated tests
setlocal
cd /d "%~dp0"
set PY=".venv\Scripts\python.exe"
set CMD=%1
if "%CMD%"=="" set CMD=status
if /i "%CMD%"=="ui"      ( %PY% scripts\webui.py start & goto :eof )
if /i "%CMD%"=="ui-stop" ( %PY% scripts\webui.py stop & goto :eof )
if /i "%CMD%"=="remote"  ( %PY% scripts\webui.py start --no-browser & "C:\Program Files\Tailscale\tailscale.exe" serve --bg 8765 & goto :eof )
if /i "%CMD%"=="remote-off" ( "C:\Program Files\Tailscale\tailscale.exe" serve --https=443 off & goto :eof )
if /i "%CMD%"=="tests"   ( %PY% -m pytest tests -q & goto :eof )
if /i "%CMD%"=="help"    ( findstr /b "rem   mvctl" "%~f0" & goto :eof )
%PY% scripts\train_plan.py %CMD%
