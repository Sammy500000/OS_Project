@echo off
cd /d "%~dp0"
python -m power_agent %*
if "%~1"=="" goto :eof
pause
