@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
 echo Primero debes ejecutar EJECUTAR_PRIMERA_PRUEBA.bat
 pause
 exit /b 1
)
.venv\Scripts\python.exe -m streamlit run app/app.py
pause
