@echo off
REM ===== PathSnap UI (Laptop 2) =====
REM Usage: run_ui.bat 192.168.x.x     (IP of the API laptop)
cd /d "%~dp0"
if exist .venv\Scripts\activate.bat call .venv\Scripts\activate.bat

set API_IP=%~1
if "%API_IP%"=="" set /p API_IP=Enter the API laptop IP (e.g. 192.168.43.10):
set PATHSNAP_API=http://%API_IP%:8000

echo Checking API at %PATHSNAP_API% ...
curl -s -m 5 %PATHSNAP_API%/health || echo WARNING: API not reachable. Check same network, firewall on Laptop 1, and the IP.
echo.
python -m streamlit run ui/app.py
pause
