@echo off
cd /d "%~dp0"
python -m pytest -q
pause
