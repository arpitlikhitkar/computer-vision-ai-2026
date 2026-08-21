@echo off
title Computer Vision AI - Phase 2 Person Tracking
echo Starting Person Tracking Demo...
cd /d "%~dp0"
.venv\Scripts\python.exe src/person_tracking.py
pause
