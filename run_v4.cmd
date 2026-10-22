@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" optimize_v4.py all
if errorlevel 1 goto end
".venv\Scripts\python.exe" v4_report.py
:end
pause
