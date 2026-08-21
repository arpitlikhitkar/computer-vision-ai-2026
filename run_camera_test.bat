@echo off
title Computer Vision AI - Phase 1 Camera Test
echo Starting Camera Test...
cd /d "%~dp0"
.venv\Scripts\python.exe src/camera_test.py
pause
