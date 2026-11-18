@echo off
REM ===== PathSnap UI - CLASSIC dark version (fallback) =====
REM Usage: run_ui_classic.bat 192.168.x.x     (IP of the API laptop)
cd /d "%~dp0"
if exist .venv\Scripts\activate.bat call .venv\Scripts\activate.bat

set API_IP=%~1
if "%API_IP%"=="" set /p API_IP=Enter the API laptop IP (e.g. 192.168.43.10):
set PATHSNAP_API=http://%API_IP%:8000

python -m streamlit run ui/app_classic.py --theme.base dark --theme.primaryColor "#a855f7" --theme.backgroundColor "#0b1020" --theme.secondaryBackgroundColor "#121a2f" --theme.textColor "#e2e8f0"
pause
