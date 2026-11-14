@echo off
cd /d "%~dp0"
echo --- PASO 1: comprobar Python ---
py -3.11 --version
if errorlevel 1 (
 echo Debes instalar Python 3.11 o modificar este archivo para tu version.
 pause
 exit /b 1
)
py -3.11 -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if errorlevel 1 goto error
python scripts/descargar_datos.py
if errorlevel 1 goto error
python entrenar.py
if errorlevel 1 goto error
python -m streamlit run app/app.py
exit /b 0
:error
echo Hubo un error. Comparte una captura de esta ventana para revisarlo.
pause
exit /b 1
