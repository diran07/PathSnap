@echo off
REM ===== Run ONCE on Laptop 1 as Administrator (right-click -> Run as administrator) =====
REM Opens TCP port 8000 so the UI laptop can reach the API.
net session >nul 2>&1
if errorlevel 1 (
  echo Please right-click this file and choose "Run as administrator".
  pause
  exit /b 1
)
netsh advfirewall firewall delete rule name="PathSnap API" >nul 2>&1
netsh advfirewall firewall add rule name="PathSnap API" dir=in action=allow protocol=TCP localport=8000
echo Done. Port 8000 is open for incoming connections.
echo To remove later: netsh advfirewall firewall delete rule name="PathSnap API"
pause
