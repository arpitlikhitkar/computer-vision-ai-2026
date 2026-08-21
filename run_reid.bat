@echo off
title Computer Vision AI - Phase 4 Person Re-ID
echo Starting Person Re-Identification (Re-ID) Demo...
cd /d "%~dp0"
.venv\Scripts\python.exe src/person_reid.py
pause
