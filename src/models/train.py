import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import os


def train_model(input_csv, model_path):
    # 1. Загружаем данные
    df = pd.read_csv(input_csv)

    # 2. Отбираем признаки (фичи)
    # Нам нужны только показатели датчиков (s_1, s_2...) и настройки (setting_)
    # Мы НЕ берем unit_nr, time_cycles и RUL, иначе модель будет "подглядывать" в ответ
    features = [c for c in df.columns if c.startswith(
        's_') or c.startswith('setting_')]
    X = df[features]
    y = df['label']

    # 3. Разделяем на обучающую и тестовую выборки (80% / 20%)
    # stratify=y важен, так как поломок (1) намного меньше, чем обычной работы (0)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Начинаем обучение на {len(X_train)} примерах...")

    # 4. Настройка CatBoost
    model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        eval_metric='AUC',
        verbose=100,      # печатать прогресс каждые 100 шагов
        random_seed=42
    )

    # Обучаем
    model.fit(
        X_train, y_train,
        eval_set=(X_test, y_test),
        early_stopping_rounds=50  # прекратить, если качество перестанет расти
    )

    # 5. Проверка качества
    preds = model.predict(X_test)
    print("\n--- ОТЧЕТ ПО МЕТРИКАМ ---")
    print(classification_report(y_test, preds))

    # 6. Сохранение артефакта (модели) во внешнюю папку
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save_model(model_path)
    print(f"\nГотовая модель сохранена в: {model_path}")


if __name__ == "__main__":
    train_model('data/processed/train_labeled.csv', 'models/engine_model.cbm')
