import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import os


def train_model(input_csv, model_path):
    # 1. Загружаем данные
    df = pd.read_csv(input_csv)

    # 2. Отбираем признаки (фичи)

    features = [c for c in df.columns if c.startswith(
        's_') or c.startswith('setting_')]
    X = df[features]
    y = df['label']

    # 3. Разделяем на обучающую и тестовую выборки (80% / 20%)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Начинаем обучение на {len(X_train)} примерах...")

    model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        eval_metric='AUC',
        verbose=100,
        random_seed=42
    )

    model.fit(
        X_train, y_train,
        eval_set=(X_test, y_test),
        early_stopping_rounds=50
    )

    # 5. Проверка качества
    preds = model.predict(X_test)
    print("\n--- ОТЧЕТ ПО МЕТРИКАМ ---")
    print(classification_report(y_test, preds))

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save_model(model_path)
    print(f"\nГотовая модель сохранена в: {model_path}")


if __name__ == "__main__":
    train_model('data/processed/train_labeled.csv', 'models/engine_model.cbm')
