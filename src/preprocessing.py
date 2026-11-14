"""Lectura y preparación reproducible del conjunto APS de Scania."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

COLUMNA_OBJETIVO = 'class'


def leer_scania(ruta: str | Path) -> pd.DataFrame:
    """Los CSV oficiales incluyen un encabezado textual de aprox. 20 líneas."""
    ruta = Path(ruta)
    if not ruta.exists():
        raise FileNotFoundError(f'No existe {ruta}. Ejecuta scripts/descargar_datos.py')
    for salto in (20, 19, 21, 0):
        try:
            datos = pd.read_csv(ruta, skiprows=salto, na_values=['na', 'NA', ''], low_memory=False)
            datos.columns = datos.columns.str.strip().str.strip('"').str.strip("'")
            if COLUMNA_OBJETIVO in datos.columns and len(datos) >= 500:
                datos[COLUMNA_OBJETIVO] = datos[COLUMNA_OBJETIVO].astype(str).str.strip().str.strip('"').str.strip("'")
                if set(datos[COLUMNA_OBJETIVO].unique()).issubset({'pos', 'neg'}):
                    for nombre in datos.columns:
                        if nombre != COLUMNA_OBJETIVO:
                            datos[nombre] = pd.to_numeric(datos[nombre], errors='coerce')
                    return datos
        except (ValueError, pd.errors.ParserError):
            continue
    raise ValueError('Formato de CSV inesperado. Verifica que descargaste los CSV originales de UCI.')


def diagnostico_datos(datos: pd.DataFrame) -> dict:
    variables = datos.drop(columns=COLUMNA_OBJETIVO)
    nulos = variables.isna().sum()
    return {
        'registros': int(len(datos)),
        'variables': int(variables.shape[1]),
        'nulos_total': int(nulos.sum()),
        'columnas_con_nulos': int((nulos > 0).sum()),
        'duplicados': int(datos.duplicated().sum()),
        'filas_con_algun_nulo': int(variables.isna().any(axis=1).sum()),
        'positivos': int((datos[COLUMNA_OBJETIVO] == 'pos').sum()),
        'negativos': int((datos[COLUMNA_OBJETIVO] == 'neg').sum()),
        'nulos_por_variable': {k: int(v) for k, v in nulos.sort_values(ascending=False).head(12).items()},
    }


class QuitarColumnasMuyVacias(BaseEstimator, TransformerMixin):
    """Filtra columnas según porcentaje de nulos aprendido en entrenamiento."""

    def __init__(self, max_porcentaje_nulos: float = 0.50):
        self.max_porcentaje_nulos = max_porcentaje_nulos

    def fit(self, X, y=None):
        self.columnas_usar_ = X.columns[X.isna().mean() < self.max_porcentaje_nulos].tolist()
        if not self.columnas_usar_:
            raise ValueError('Todas las columnas fueron descartadas. Revisa los datos originales.')
        return self

    def transform(self, X):
        return X.loc[:, self.columnas_usar_].copy()
