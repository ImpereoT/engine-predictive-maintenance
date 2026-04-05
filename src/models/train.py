import pandas as pd
import os
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, f1_score


def train_model(input_csv, model_path):
    # 1. Загружаем данные
    df = pd.read_csv(input_csv)

    # 2. Отбираем признаки
    features = [c for c in df.columns if c.startswith(
        's_') or c.startswith('setting_')]
    X = df[features]
    y = df['label']

    # 3. Разделяем на train/test (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Начинаем обучение на {len(X_train)} примерах...")

    # 4. Обучаем модель
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
    proba = model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, proba)
    f1 = f1_score(y_test, preds)

    print("\n--- ОТЧЕТ ПО МЕТРИКАМ ---")
    print(classification_report(y_test, preds))
    print(f"ROC-AUC:  {roc_auc:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"Итого примеров в тесте: {len(y_test)}")
    print(f"Дефектных двигателей:   {y_test.sum()} ({y_test.mean()*100:.1f}%)")

    # 6. Сохраняем модель
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save_model(model_path)
    print(f"\nГотовая модель сохранена в: {model_path}")


if __name__ == "__main__":
    train_model('data/processed/train_labeled.csv', 'models/engine_model.cbm')
