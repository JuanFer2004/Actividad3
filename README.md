# PCA aplicado a fallas en camiones Scania — Asesoría de avances

Proyecto académico de Inteligencia Artificial I (Actividad 3, Entrega 1).

**Problema.** A partir de señales operativas reales y anonimizadas de camiones Scania, clasificar si un fallo está relacionado con el sistema de presión de aire (APS). El conjunto no permite diagnosticar una avería concreta en un camión particular. El sistema solo clasifica registros en la categoría `pos` (fallo relacionado con APS) o `neg` (fallo en otro componente).

**Método principal:** análisis de componentes principales (**PCA**). PCA no es un clasificador: reduce las dimensiones de los indicadores tras limpiarlos y estandarizarlos. Un clasificador de regresión logística usa después las componentes principales para hacer la predicción. **No** se usa Random Forest, Titanic ni datos sintéticos.

**Datos públicos REALES:** [APS Failure at Scania Trucks — UCI](https://archive.ics.uci.edu/dataset/421/aps+failure+at+scania+trucks) (DOI 10.24432/C51S51). Entrenamiento oficial: 60.000 registros, 170 atributos de entrada; 59.000 negativos y 1.000 positivos. Archivo de prueba independiente: 16.000 registros. Los datos reales incluyen numerosos valores faltantes indicados por `na`. Sus columnas están anonimizadas por Scania, por lo que sería incorrecto atribuirles nombres de sensores específicos.

## ¿Qué se hace con los datos?

1. Se descargan los dos CSV originales de UCI, sin inventar datos.
2. Se revisan número de registros, valores faltantes, variables y filas duplicadas completas.
3. Se divide estratificadamente el archivo oficial de entrenamiento en 80 % entrenamiento y 20 % validación.
4. En los registros de **entrenamiento**, se aprenden qué columnas tienen menos del 50 % de ausencias. En las columnas conservadas, los faltantes se imputan con la mediana y las variables se estandarizan. Esto se hace en un `Pipeline` para evitar filtrar información de validación.
5. PCA reduce el conjunto a **20 componentes principales**; se calcula cuánta varianza explican realmente.
6. Una regresión logística con ponderación de clases clasifica a partir de esas 20 componentes. Se informan accuracy, precision, recall, F1, ROC AUC y costo de equivocaciones (10 por falsa alarma y 500 por omitir falla APS).
7. Se guarda el pipeline **entero** en `models/modelo_pca_scania.joblib`, junto con las métricas reales en `models/metricas.json`.
8. Se usa una aplicación Streamlit local que permite elegir casos REALES del archivo de prueba y cambiar al menos 3 indicadores anonimizados para explorar escenarios.

## Instrucciones fáciles para Windows (primera ejecución)

Requisito: **Python 3.11** instalado desde https://www.python.org/downloads/ (en el instalador marcar **Add Python to PATH**). Requiere internet **la primera vez**, para instalar paquetes y descargar los CSV públicos (~54 MB comprimidos). No necesita AWS ni nube en la primera asesoría.

1. Descomprime el ZIP del proyecto en tu PC, por ejemplo en `Documentos\pca-camiones`.
2. Haz doble clic en **`EJECUTAR_PRIMERA_PRUEBA.bat`**.
3. El programa instalará dependencias, descargará datos auténticos de UCI, limpiará los datos, entrenará PCA y el clasificador, creará el `.joblib` y abrirá una página de Streamlit (habitualmente `http://localhost:8501`).
4. En la aplicación selecciona *Ejemplo A*, pulsa **Analizar registro** y toma una captura. Repite con *Ejemplo B*.
5. Abre la sección **¿Qué limpieza de datos se hizo?** y toma una captura que muestre las cifras reales.
6. Abre `docs/INFORME_AVANCE_REAL.txt`, generado **después** del entrenamiento, para copiar cifras reales a tu informe.
7. Las veces siguientes puedes abrir **`INICIAR_EN_WINDOWS.bat`** para iniciar la app sin reentrenar.

Si tu versión instalada de Python no es 3.11, puedes usar la ejecución manual indicada abajo con `py -3.12` o la versión compatible que tengas.

### Opción manual desde PowerShell / terminal de VS Code

Entra en la carpeta principal y ejecuta, uno a uno:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python scripts/descargar_datos.py
python entrenar.py
python -m streamlit run app/app.py
```

> Si PowerShell bloquea la activación del entorno, usa **Símbolo del sistema (CMD)** en vez de PowerShell y escribe `.venv\Scripts\activate.bat`. O ejecuta directamente `.venv\Scripts\python.exe` sin activar el entorno.

### Si falla la descarga automática

Abre la [página oficial de UCI](https://archive.ics.uci.edu/dataset/421/aps+failure+at+scania+trucks), descarga el ZIP original y extrae estos dos archivos en `data/raw/`:

- `aps_failure_training_set.csv`
- `aps_failure_test_set.csv`

Luego ejecuta `python entrenar.py`. No edites ni reemplaces sus datos con ejemplos inventados.

## Carpeta del proyecto

```text
pca-camiones/
├── README.md                        # este instructivo
├── .gitignore
├── requirements.txt
├── EJECUTAR_PRIMERA_PRUEBA.bat      # descarga + entrena + abre app
├── INICIAR_EN_WINDOWS.bat          # abre app tras entrenamiento
├── entrenar.py                      # entrenamiento y métricas
├── scripts/
│   └── descargar_datos.py           # fuente real UCI
├── src/
│   ├── __init__.py
│   └── preprocessing.py             # lectura, diagnóstico y limpieza
├── data/
│   ├── raw/                          # CSV originales (se descargan)
│   └── processed/                    # reservado para trabajo futuro
├── models/
│   ├── modelo_pca_scania.joblib      # SE GENERA tras ejecutar entrenar.py
│   └── metricas.json                # SE GENERA tras entrenar.py
├── notebooks/
│   └── 01_training.ipynb            # notebook de exploración y aprendizaje
├── app/
│   └── app.py                        # interfaz local Streamlit
└── docs/
    ├── EVIDENCIAS_PARA_EL_PROFESOR.md
    ├── SUBIR_A_GITHUB.md
    ├── GRUPO_Actividad3_ASESORIA.txt # plantilla inicial, completar
    └── INFORME_AVANCE_REAL.txt       # SE GENERA con los datos auténticos
```

## Primera entrega: qué se presentará

- [ ] **Modelo entrenado y serializado:** ejecutar entrenamiento y verificar que existe `.joblib` y que la app lo carga.
- [x] **Código de la app en desarrollo:** `app/app.py`, con interfaz y predicciones.
- [ ] **Repositorio GitHub público con estructura básica:** subir archivos con el equipo, después de realizar la prueba.
- [ ] **Evidencia real de prueba local:** capturas del PC del estudiante, después de ejecutar la aplicación.

## Limitaciones y buenas prácticas

- **No hay métricas inventadas:** accuracy, recall y varianza PCA se calculan cuando se ejecuta el entrenamiento en datos reales.
- PCA reduce dimensiones, pero la predicción depende del clasificador posterior. Atribuir un resultado al PCA solo sería incorrecto.
- La clase `pos` es minoritaria (~1,7 % de entrenamiento); el accuracy por sí solo no basta. Se prioriza también **recall** y el costo de no detectar una falla.
- No se manipulan ni se imputan etiquetas de clase. Los datos faltantes de variables se imputan **después** de separar entrenamiento y validación.
- **Datos originales y modelo:** las versiones originales de los archivos CSV se mantienen sin cambios en `data/raw/`. El pipeline aplica transformaciones sin alterar esos archivos.
- Las métricas y las capturas solo son evidencias **después** de ejecutar los scripts. Una captura de esta descripción no sustituye evidencia del PC.

## Repositorio

https://github.com/JuanFer2004/Actividad3

## Autores

Agregar los nombres de los tres integrantes del grupo. Cada integrante debe hacer aportes reales al repositorio.

## Referencia

Scania, *APS Failure at Scania Trucks*, UCI Machine Learning Repository, DOI: 10.24432/C51S51. [Fuente oficial](https://archive.ics.uci.edu/dataset/421/aps+failure+at+scania+trucks).