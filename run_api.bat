@echo off
REM ===== PathSnap API (Laptop 1) =====
REM Usage: run_api.bat [path\to\evaluator_trips]
REM With no argument, uses data\raw\student_test.
cd /d "%~dp0"
if exist .venv\Scripts\activate.bat call .venv\Scripts\activate.bat

if not "%~1"=="" set PATHSNAP_EVAL_DIR=%~1

echo.
echo ==== This laptop's IP addresses ====
echo Give the UI laptop the IP of your Wi-Fi adapter (NOT VirtualBox / vEthernet / Loopback):
powershell -NoProfile -Command "Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.IPAddress -notlike '127.*' -and $_.IPAddress -notlike '169.254.*' } | Sort-Object { $_.InterfaceAlias -notmatch 'Wi-?Fi|WLAN|Ethernet' } | ForEach-Object { '  {0,-16} {1}' -f $_.IPAddress, $_.InterfaceAlias }"
echo.
if defined PATHSNAP_EVAL_DIR (echo Eval trips folder: %PATHSNAP_EVAL_DIR%) else (echo Eval trips folder: data\raw\student_test)
echo API will listen on port 8000. Keep this window open - request logs appear here.
echo.
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
pause
