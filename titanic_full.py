# titanic_full.py
# Полный скрипт для анализа Titanic (логистическая регрессия с предобработкой и GridSearch)
# Запуск: python titanic_full.py
# Требования: pandas, numpy, matplotlib, seaborn, scikit-learn, joblib

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, roc_auc_score, classification_report
)
import joblib

# -------------------------
# Настройки вывода и директории
# -------------------------
OUTPUT_DIR = "titanic_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
plt.rcParams.update({'figure.max_open_warning': 0})

# -------------------------
# 1) Загрузка данных
# -------------------------
def load_data():
    """
    Загружает датасет Titanic через seaborn.
    Если у вас локальный CSV, замените на pd.read_csv('path/to/file.csv').
    """
    df = sns.load_dataset('titanic')
    return df

# -------------------------
# 2) Базовый EDA (печать краткой информации)
# -------------------------
def basic_eda(df):
    print("Форма данных:", df.shape)
    print("\nПервые 5 строк:")
    display = getattr(pd, "set_option", None)
    print(df.head(5).to_string(index=False))
    print("\nТипы столбцов и пропуски:")
    print(df.info())
    print("\nСумма пропусков по столбцам:")
    print(df.isnull().sum())

# -------------------------
# 3) Предобработка и инженерия признаков
# -------------------------
def prepare_features(df):
    """
    Выбираем признаки, обрабатываем пропуски и готовим списки признаков.
    Возвращаем X (DataFrame) и y (Series).
    """
    df = df.copy()
    # Удаляем столбцы, которые дублируют информацию или малоинформативны:
    drop_cols = ['alive', 'class', 'embark_town', 'who', 'adult_male', 'alone', 'deck']
    for c in drop_cols:
        if c in df.columns:
            df.drop(columns=[c], inplace=True)

    # Целевая переменная
    if 'survived' not in df.columns:
        raise ValueError("Ожидаемый столбец 'survived' не найден в данных")

    # Определим числовые и категориальные колонки для модели
    # В списке можно добавлять/убирать признаки по желанию
    num_cols = ['age', 'fare', 'sibsp', 'parch']
    # Убедимся, что они есть в df
    num_cols = [c for c in num_cols if c in df.columns]

    cat_cols = []
    for c in ['sex', 'pclass', 'embarked']:
        if c in df.columns:
            cat_cols.append(c)

    X = df.drop(columns=['survived'])
    y = df['survived'].astype(int)

    return X, y, num_cols, cat_cols

# -------------------------
# 4) Построение pipeline и подбор гиперпараметров
# -------------------------
def build_and_search_model(X_train, y_train, num_cols, cat_cols):
    # Трансформеры
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, num_cols),
        ('cat', categorical_transformer, cat_cols)
    ], remainder='drop')

    # Полный pipeline
    pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('clf', LogisticRegression(solver='liblinear', max_iter=2000))
    ])

    # Сетка гиперпараметров
    param_grid = {
        'clf__C': [0.01, 0.1, 1, 10, 100],
        'clf__penalty': ['l2']
    }

    grid = GridSearchCV(pipe, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=0)
    grid.fit(X_train, y_train)
    return grid

# -------------------------
# 5) Оценка модели и визуализации
# -------------------------
def evaluate_and_plot(model, X_test, y_test, num_cols, cat_cols):
    # Предсказания
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Метрики
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    rocauc = roc_auc_score(y_test, y_proba)

    metrics = {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'roc_auc': rocauc
    }

    print("\n=== Метрики на тестовой выборке ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Матрица ошибок
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Pred 0','Pred 1'], yticklabels=['True 0','True 1'])
    plt.xlabel('Предсказано')
    plt.ylabel('Истинно')
    plt.title('Матрица ошибок')
    cm_path = os.path.join(OUTPUT_DIR, 'confusion_matrix.png')
    plt.tight_layout()
    plt.savefig(cm_path, dpi=150)
    # plt.close() # Remove plt.close() to display the plot in the notebook
    print(f"Матрица ошибок сохранена: {cm_path}")

    # ROC-кривая
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0,1], [0,1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC-кривая')
    plt.legend(loc='lower right')
    roc_path = os.path.join(OUTPUT_DIR, 'roc_curve.png')
    plt.tight_layout()
    plt.savefig(roc_path, dpi=150)
    # plt.close() # Remove plt.close() to display the plot in the notebook
    print(f"ROC-кривая сохранена: {roc_path}")

    # Важность признаков (коэффициенты)
    # Получим имена после one-hot кодирования
    preprocessor = model.best_estimator_.named_steps['preprocessor']
    # numeric names
    numeric_names = num_cols
    # categorical OHE names
    try:
        ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
        cat_feature_names = list(ohe.get_feature_names_out(cat_cols))
    except Exception:
        # Если нет категориальных или был другой тип трансформера
        cat_feature_names = []
    feature_names = numeric_names + cat_feature_names

    coefs = model.best_estimator_.named_steps['clf'].coef_[0]
    # защита: длина имен должна совпадать с длиной коэф.
    if len(feature_names) != len(coefs):
        # Попробуем восстановить правильно: если ColumnTransformer возвращает другую порядковость.
        # Лучше — получить feature_names_out (sklearn >=1.0), но не все версии поддерживают.
        # На крайний случай покажем первые N
        feature_names = [f"f{i}" for i in range(len(coefs))]

    feat_imp = pd.Series(coefs, index=feature_names).sort_values()
    plt.figure(figsize=(7,6))
    feat_imp.plot(kind='barh')
    plt.title('Коэффициенты логистической регрессии (влияние признаков)')
    plt.xlabel('Коэффициент')
    plt.tight_layout()
    feat_path = os.path.join(OUTPUT_DIR, 'feature_coefs.png')
    plt.savefig(feat_path, dpi=150)
    # plt.close() # Remove plt.close() to display the plot in the notebook
    print(f"Коэффициенты признаков сохранены: {feat_path}")

    # Сохраняем метрики в CSV
    metrics_df = pd.DataFrame([metrics])
    metrics_csv = os.path.join(OUTPUT_DIR, 'metrics.csv')
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"Метрики сохранены: {metrics_csv}")

    return metrics, {
        'confusion_matrix': cm_path,
        'roc_curve': roc_path,
        'feature_coefs': feat_path,
        'metrics_csv': metrics_csv
    }

# -------------------------
# main
# -------------------------
def main():
    df = load_data()
    print("=== Базовый обзор данных ===")
    basic_eda(df)

    X, y, num_cols, cat_cols = prepare_features(df)
    print("\nЧисловые признаки:", num_cols)
    print("Категориальные признаки:", cat_cols)

    # Разделение выборки
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    print("\nРазмеры: X_train, X_test:", X_train.shape, X_test.shape)

    # Подбор модели
    print("\nЗапуск GridSearchCV (pipeline + logistic regression)...")
    grid = build_and_search_model(X_train, y_train, num_cols, cat_cols)
    print("Лучшие параметры GridSearch:", grid.best_params_)
    print(f"Лучший CV ROC AUC: {grid.best_score_:.4f}")

    # Оценка
    metrics, saved_paths = evaluate_and_plot(grid, X_test, y_test, num_cols, cat_cols)

    # Сохранение модели
    model_path = os.path.join(OUTPUT_DIR, 'model_joblib.pkl')
    joblib.dump(grid, model_path)
    print(f"\nПолный GridSearchCV (включая pipeline) сохранён: {model_path}")

    print("\nГотово. Файлы сохранены в папке:", OUTPUT_DIR)
    print("Список сохранённых файлов:")
    for k, v in saved_paths.items():
        print(f" - {k}: {v}")
    print(f" - model: {model_path}")

    # List files in the output directory
    print(f"\nСодержимое директории '{OUTPUT_DIR}':")
    for filename in os.listdir(OUTPUT_DIR):
        print(f"- {filename}")


if __name__ == "__main__":
    main()