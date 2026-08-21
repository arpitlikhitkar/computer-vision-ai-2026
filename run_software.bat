@echo off
TITLE Household AI — Security & Face Recognition Web Software
cd /d "%~dp0"
echo ==================================================
echo  Starting Household AI Web Software Engine...
echo  Opening Web Dashboard at http://127.0.0.1:5000
echo ==================================================
start http://127.0.0.1:5000
.venv\Scripts\python.exe app.py
pause
