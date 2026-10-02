@echo off
cd /d "%~dp0"
python --version >nul 2>&1
if errorlevel 1 (
    echo Erreur : Python n'est pas installe ou pas dans le PATH.
    echo Telechargez-le sur https://www.python.org/downloads/
    pause
    exit /b 1
)
python beaver.py %*
if errorlevel 1 pause
exit /b %errorlevel%
