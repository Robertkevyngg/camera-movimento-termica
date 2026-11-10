@echo off
rem Inicia o aplicativo desktop no Windows usando o ambiente virtual do projeto.
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" desktop_app.py %*
) else (
    python desktop_app.py %*
)
