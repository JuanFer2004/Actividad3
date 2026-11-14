"""Descarga los datos públicos y REALES de Scania APS desde UCI."""
from __future__ import annotations
from pathlib import Path
import io
import sys
import zipfile
import requests

RAIZ = Path(__file__).resolve().parents[1]
DESTINO = RAIZ / 'data' / 'raw'
FUENTE = 'https://archive.ics.uci.edu/static/public/421/aps%2Bfailure%2Bat%2Bscania%2Btrucks.zip'
ALTERNATIVA = 'https://archive.ics.uci.edu/static/public/421/aps+failure+at+scania+trucks.zip'
NOMBRES = ('aps_failure_training_set.csv', 'aps_failure_test_set.csv')


def descargar():
    DESTINO.mkdir(parents=True, exist_ok=True)
    if all((DESTINO / n).exists() for n in NOMBRES):
        print('Los dos archivos originales ya están descargados.')
        return
    ultima_excepcion = None
    for url in (FUENTE, ALTERNATIVA):
        try:
            print('Descargando archivo original desde UCI...')
            respuesta = requests.get(url, timeout=180, headers={'User-Agent': 'PCA-Camiones-Actividad-Academica/1.0'})
            respuesta.raise_for_status()
            archivo = zipfile.ZipFile(io.BytesIO(respuesta.content))
            for nombre in NOMBRES:
                coincidencias = [m for m in archivo.namelist() if m.lower().endswith(nombre)]
                if not coincidencias:
                    raise FileNotFoundError(f'No se encontró {nombre} en el ZIP oficial')
                datos = archivo.read(coincidencias[0])
                (DESTINO / nombre).write_bytes(datos)
                print(f'Guardado: {nombre} ({len(datos):,} bytes)')
            print('Descarga finalizada. Fuente: UCI, dataset 421.')
            return
        except (requests.RequestException, zipfile.BadZipFile, FileNotFoundError) as e:
            ultima_excepcion = e
    print('No fue posible descargar el dataset automáticamente.', file=sys.stderr)
    print(f'Detalle: {ultima_excepcion}', file=sys.stderr)
    print('Alternativa: abrir https://archive.ics.uci.edu/dataset/421/aps+failure+at+scania+trucks', file=sys.stderr)
    print('Descargar el ZIP y copiar los dos archivos CSV en data/raw/', file=sys.stderr)
    raise SystemExit(1)


if __name__ == '__main__':
    descargar()
