# Evidencias pendientes de la primera entrega

Las capturas deben hacerse de una ejecución real del proyecto en el computador del equipo, no deben simularse.

1. **Datos antes de limpieza.** Capturar salida de CMD de `python entrenar.py` con registros, variables, valores nulos y duplicados.
2. **Entrenamiento.** Capturar métricas y varianza explicada después de finalizar el entrenamiento.
3. **Modelo serializado.** Capturar `models/modelo_pca_scania.joblib` y `models/metricas.json` en el explorador.
4. **Prueba local.** Capturar Streamlit en `http://localhost:8501` y analizar el Ejemplo A.
5. **Segunda predicción.** Analizar el Ejemplo B y capturar el resultado.
6. **Limpieza.** Abrir la sección de limpieza de Streamlit y mostrar datos reales de ausencias.
7. **Repositorio público.** Capturar https://github.com/JuanFer2004/Actividad3 con sus archivos.

**Atención:** Una predicción incorrecta no debe ocultarse. El objetivo es mostrar un prototipo y explicar la evaluación honesta.

## Guion para el profesor

Profesor, estamos desarrollando una aplicación para clasificar si las fallas registradas de camiones Scania están relacionadas con su sistema de presión de aire. Trabajamos con datos reales de UCI y con múltiples variables que incluyen ausencias. El enfoque central es PCA, con el que reducimos la cantidad de variables después de limpiarlas y estandarizarlas. Para obtener una predicción, usamos un clasificador logístico sobre las componentes. Durante la asesoría mostraremos el modelo entrenado y sus métricas reales, el archivo serializado y una prueba en Streamlit. El despliegue público quedará para la siguiente entrega.

**El guion en pasado o presente de logro solo puede utilizarse cuando se haya ejecutado la prueba.**
