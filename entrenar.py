"""Entrenamiento reproducible: limpieza, PCA y regresión logística con datos de Scania."""
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.preprocessing import leer_scania, diagnostico_datos, QuitarColumnasMuyVacias

RAIZ = Path(__file__).resolve().parent


def main():
    ruta = RAIZ / 'data' / 'raw' / 'aps_failure_training_set.csv'
    datos = leer_scania(ruta)
    reporte = diagnostico_datos(datos)
    print('\n=== INFORME DE DATOS REALES ANTES DE LA LIMPIEZA ===')
    for k, v in reporte.items():
        if k != 'nulos_por_variable':
            print(f'{k}: {v}')
    print('Variables con más valores faltantes:', reporte['nulos_por_variable'])
    X = datos.drop(columns='class')
    y = (datos['class'] == 'pos').astype(int)
    X_entrena, X_valida, y_entrena, y_valida = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    modelo = Pipeline([
        ('eliminar_columnas', QuitarColumnasMuyVacias(0.50)),
        ('rellenar_nulos', SimpleImputer(strategy='median', keep_empty_features=True)),
        ('estandarizar', StandardScaler()),
        ('pca', PCA(n_components=20, svd_solver='randomized', random_state=42)),
        ('clasificar', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)),
    ])
    print('\nEntrenando modelo con registros auténticos de Scania...')
    modelo.fit(X_entrena, y_entrena)
    pred = modelo.predict(X_valida)
    prob = modelo.predict_proba(X_valida)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_valida, pred, labels=[0, 1]).ravel()
    metricas = {
        'accuracy': float(accuracy_score(y_valida, pred)),
        'precision': float(precision_score(y_valida, pred, zero_division=0)),
        'recall': float(recall_score(y_valida, pred, zero_division=0)),
        'f1': float(f1_score(y_valida, pred, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_valida, prob)),
        'verdaderos_negativos': int(tn),
        'falsos_positivos': int(fp),
        'falsos_negativos': int(fn),
        'verdaderos_positivos': int(tp),
        'costo_errores_uci': int(fp * 10 + fn * 500),
    }
    columnas = modelo.named_steps['eliminar_columnas'].columnas_usar_
    pca = modelo.named_steps['pca']
    metadata = {
        'fecha_entrenamiento_utc': datetime.now(timezone.utc).isoformat(),
        'fuente': 'https://archive.ics.uci.edu/dataset/421/aps+failure+at+scania+trucks',
        'dataset_original': reporte,
        'registros_entrenamiento': int(len(X_entrena)),
        'registros_validacion': int(len(X_valida)),
        'variables_antes': int(X.shape[1]),
        'variables_despues_filtro': len(columnas),
        'columnas_originales': list(X.columns),
        'columnas_utilizadas': columnas,
        'componentes_pca': int(pca.n_components_),
        'varianza_explica_pca': float(np.sum(pca.explained_variance_ratio_)),
        'metricas_validacion': metricas,
    }
    (RAIZ / 'models').mkdir(parents=True, exist_ok=True)
    joblib.dump(modelo, RAIZ / 'models' / 'modelo_pca_scania.joblib', compress=3)
    (RAIZ / 'models' / 'metricas.json').write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8'
    )
    informe = [
        'ASESORIA DE AVANCES - Actividad 3 - PCA',
        'Proyecto: Clasificación de fallas APS en camiones Scania',
        'Fuente: UCI Machine Learning Repository, dataset 421',
        'Algoritmo principal: PCA (20 componentes) + regresión logística',
        f"Registros originales: {reporte['registros']}",
        f"Variables originales: {reporte['variables']}",
        f"Celdas con valores faltantes: {reporte['nulos_total']}",
        f"Filas con al menos un faltante: {reporte['filas_con_algun_nulo']}",
        f"Duplicados completos detectados: {reporte['duplicados']}",
        f"Variables retenidas tras filtrar nulos: {len(columnas)}",
        f"Componentes PCA: {int(pca.n_components_)}",
        f"Varianza explicada acumulada: {metadata['varianza_explica_pca']:.2%}",
        f"Accuracy validación: {metricas['accuracy']:.2%}",
        f"Precision fallas APS: {metricas['precision']:.2%}",
        f"Recall fallas APS: {metricas['recall']:.2%}",
        f"F1 fallas APS: {metricas['f1']:.2%}",
        f"ROC AUC: {metricas['roc_auc']:.3f}",
        f"Matriz confusión (TN,FP,FN,TP): {tn}, {fp}, {fn}, {tp}",
        f"Costo de errores (10 x FP + 500 x FN): {metricas['costo_errores_uci']}",
        'Archivo generado: models/modelo_pca_scania.joblib',
        'Aplicación: app/app.py (Streamlit)',
        'Repositorio GitHub: https://github.com/JuanFer2004/Actividad3',
        'Capturas del PC: PENDIENTE de agregar tras prueba local',
    ]
    (RAIZ / 'docs').mkdir(parents=True, exist_ok=True)
    (RAIZ / 'docs' / 'INFORME_AVANCE_REAL.txt').write_text('\n'.join(informe)+'\n', encoding='utf-8')
    print('\n=== RESULTADOS REALES EN VALIDACIÓN ===')
    for clave, valor in metricas.items():
        print(f'{clave}: {valor}')
    print(f"PCA conservó {metadata['varianza_explica_pca']:.2%} de la varianza")
    print('Modelo guardado en models/modelo_pca_scania.joblib')
    return modelo, metadata


if __name__ == '__main__':
    main()
