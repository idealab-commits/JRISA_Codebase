@echo off
REM Whole JRISA pipeline. Examples:
REM   run_all.bat
REM   run_all.bat --dataset ac4 --configs full full_afg --seeds 0 1 2
cd /d "%~dp0"
python run_all.py %*
pause
