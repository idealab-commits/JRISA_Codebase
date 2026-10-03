@echo off
REM ======================================================================
REM  JRISA - one-click data setup
REM  Double-click this file, or run it from a terminal in this folder.
REM
REM  Default (what this file does): AC4 + MitoEM-R/H, writing only
REM      <name>_raw.npy and <name>_labels.npy
REM  into data\ac3_ac4\ and data\mitoem\, then prints a check table.
REM
REM  To also get CREMI and the _sem_gt.npy files, run instead:
REM      python scripts\prepare_data.py
REM ======================================================================
cd /d "%~dp0"

if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
)

echo Installing data packages...
python -m pip install --quiet numpy h5py huggingface_hub intern remotezip pillow
if errorlevel 1 (
    echo Package install failed. Check your internet connection and Python install.
    pause
    exit /b 1
)

python scripts\prepare_data.py --skip-cremi --no-sem-gt %*
if errorlevel 1 (
    echo.
    echo Data setup finished WITH PROBLEMS - read the messages above.
) else (
    echo.
    echo Data setup finished OK.
)
pause
