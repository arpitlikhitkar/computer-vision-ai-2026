@echo off
TITLE Household AI — PySide6 Desktop Application
cd /d "%~dp0"
set PYTHONPATH=.
echo ==================================================
echo  Starting Household AI PySide6 Desktop Application...
echo ==================================================
.venv\Scripts\python.exe -m app.main
pause
