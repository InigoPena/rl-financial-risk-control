import os
import pandas as pd


def load_dataset(name, index_name):
    # Ir desde data_loading/ hasta Codigo/data/
    base_dir = os.path.dirname(os.path.abspath(__file__))  # data_loading/
    # Subir a data_processing/, luego a src/, luego a Codigo/, luego a data/
    path = os.path.join(base_dir, '..', '..', '..', 'data', name + '.csv')
    path = os.path.normpath(path)  # Normalizar la ruta
    df = pd.read_csv(path, parse_dates=True, index_col=index_name)
    return df
