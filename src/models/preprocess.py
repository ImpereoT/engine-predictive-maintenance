import pandas as pd
import numpy as np
import os


def preprocess_data(input_path, output_path):

    index_names = ['unit_nr', 'time_cycles']
    setting_names = ['setting_1', 'setting_2', 'setting_3']
    sensor_names = [f's_{i}' for i in range(1, 22)]
    col_names = index_names + setting_names + sensor_names

    print(f"Читаем данные из {input_path}...")

    # Загружаем данные
    df = pd.read_csv(input_path, sep='\s+', header=None, names=col_names)

    # 2. Расчет RUL (Remaining Useful Life)

    max_cycles = df.groupby('unit_nr')['time_cycles'].max().reset_index()
    max_cycles.columns = ['unit_nr', 'max_cycle']

    df = df.merge(max_cycles, on='unit_nr', how='left')

    # RUL = сколько циклов осталось до поломки
    df['RUL'] = df['max_cycle'] - df['time_cycles']

    # 3. Создаем таргет (Label)

    df['label'] = (df['RUL'] <= 30).astype(int)

    df = df.drop(columns=['max_cycle'])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Успех! Файл сохранен: {output_path}")
    print(f"Всего записей: {len(df)}")
    print(f"Из них в 'зоне риска' (label=1): {df['label'].sum()}")


if __name__ == "__main__":

    RAW_DATA = 'data/raw/train_FD001.txt'
    PROCESSED_DATA = 'data/processed/train_labeled.csv'

    preprocess_data(RAW_DATA, PROCESSED_DATA)
