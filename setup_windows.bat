@echo off
cd /d "%~dp0"
echo Installing dependencies...
python -m pip install -r requirements.txt
echo.
echo Setup done. Next: run_demo.bat (simulated laptop) or run_agent.bat (your real laptop).
pause
