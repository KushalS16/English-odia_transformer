@echo off
setlocal
cd /d "%~dp0"
echo ================================================
echo Test-7 English -> Odia Transformer
 echo Windows CMD environment setup
echo ================================================
where python >nul 2>nul
if errorlevel 1 (
  echo ERROR: Python was not found. Install Python 3.10-3.12 and enable Add Python to PATH.
  exit /b 1
)
python -m venv .venv || exit /b 1
call .venv\Scripts\activate.bat || exit /b 1
python -m pip install --upgrade pip || exit /b 1
pip install -r requirements.txt || exit /b 1
python scripts\preflight.py || exit /b 1
python scripts\run_tests.py || exit /b 1
echo.
echo Environment and tests are ready.
echo Next commands:
echo   call .venv\Scripts\activate.bat
echo   python run_project.py setup
echo   python run_project.py train --fresh
echo   python run_project.py evaluate
echo   python run_project.py ui
endlocal
