@echo off
setlocal
cd /d "%~dp0"
python run_all.py
if errorlevel 1 (
  echo.
  echo Pipeline failed. Review the error above.
  pause
  exit /b 1
)
echo.
echo Version 2 pipeline completed successfully.
pause
