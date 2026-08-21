@echo off
TITLE Computer Vision AI - Household Member Enrollment
cd /d "%~dp0"
echo ==================================================
echo  Starting Household Member Enrollment System...
echo ==================================================
.venv\Scripts\python.exe src/enrollment/enroll_person.py
pause
