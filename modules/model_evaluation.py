import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                             confusion_matrix, precision_score, recall_score, f1_score, fbeta_score)


def evaluate_classification_model(model, X_test, y_test, beta=2, threshold=0.5):
    """
    Оценивает Keras модель и выводит результаты: accuracy, confusion matrix, precision, recall, F1-score и F-beta score.

    :param model: Keras Model
    :param X_test: Тестовые данные (массив или список массивов для многовходных моделей)
    :param y_test: Истинные значения
    :param beta: Значение beta для F-beta score (по умолчанию 2)
    :param threshold: Порог классификации (по умолчанию 0.5)
    :return: Tuple (results_df, annotated_cm)
    """
    # Проверка размерностей данных
    if isinstance(X_test, list):
        for i, x in enumerate(X_test):
            if len(x) != len(y_test):
                raise ValueError(
                    f"Размер входного массива X_test[{i}] ({len(x)}) не совпадает с размером y_test ({len(y_test)})."
                )
    else:
        if len(X_test) != len(y_test):
            raise ValueError(
                f"Размер входного массива X_test ({len(X_test)}) не совпадает с размером y_test ({len(y_test)})."
            )

    # Оцениваем модель
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
    print(f"Test Loss: {test_loss:.5f}")
    print(f"Test Accuracy: {test_accuracy:.5f}")

    # Предсказания вероятностей
    y_pred_proba = model.predict(X_test)

    # Применяем порог классификации
    y_pred = (y_pred_proba > threshold).astype(int).flatten()

    # Вычисление метрик
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    f_beta = fbeta_score(y_test, y_pred, beta=beta, zero_division=0)

    print("\nMetrics:")
    print(f"Precision: {precision:.5f}")
    print(f"Recall: {recall:.5f}")
    print(f"F1-score: {f1:.5f}")
    print(f"F-beta ({beta}): {f_beta:.5f}")
    print(f"Threshold used: {threshold}")

    # Создаем DataFrame с результатами
    results_df = pd.DataFrame({
        'y_test': y_test,
        'y_pred_proba': y_pred_proba.flatten(),  # Добавляем вероятности
        'y_pred': y_pred,
    })
    results_df['prediction_is_true'] = results_df.apply(
        lambda row: 'Yes' if row['y_test'] == row['y_pred'] else 'No',
        axis=1
    )

    # Выводим первые строки DataFrame
    print("\nResults DataFrame (first 5 rows):")
    print(results_df.head())

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(
        cm, 
        index=['Actual No (0)', 'Actual Yes (1)'], 
        columns=['Predicted No (0)', 'Predicted Yes (1)']
    )

    # Аннотированная confusion matrix
    annotations = [
        ['TN (True Negative)', 'FP (False Positive)'],
        ['FN (False Negative)', 'TP (True Positive)']
    ]
    annotated_cm = cm_df.astype(str)
    for i, row in enumerate(cm_df.index):
        for j, col in enumerate(cm_df.columns):
            annotated_cm.loc[row, col] = f"{annotations[i][j]}: {cm[i, j]}"

    # Выводим annotated confusion matrix
    print("\nAnnotated Confusion Matrix:")
    print(annotated_cm)

    return results_df, annotated_cm


def evaluate_regression_model(model, X_test, y_test):
    """
    Оценивает Keras модель для задачи регрессии.

    :param model: Keras Model
    :param X_test: Тестовые данные
    :param y_test: Истинные значения
    :return: results_df (с предсказаниями), метрики (MSE, MAE, R²)
    """
    # Оцениваем модель
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=1)
    print(f"Test Loss (MSE): {test_loss:.5f}")
    print(f"Test MAE: {test_mae:.5f}")

    # Предсказания модели
    y_pred = model.predict(X_test).flatten()

    # Вычисляем метрики
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\nRegression Metrics:")
    print(f"Mean Squared Error (MSE): {mse:.5f}")
    print(f"Mean Absolute Error (MAE): {mae:.5f}")
    print(f"R² Score: {r2:.5f}")

    # Создаем DataFrame с результатами
    results_df = pd.DataFrame({
        'y_test': y_test,
        'y_pred': y_pred
    })

    return results_df, mse, mae, r2


def plot_learning_curves(history, metric='mae'):
    """
    Строит кривые обучения (loss и заданная метрика) для Keras модели, чтобы оценить переобучение.
    
    :param history: объект History, возвращаемый model.fit()
    :param metric: названия метрики из history.history (например, 'accuracy', 'mae', 'f1', 'r2')
    """
    # Извлекаем loss
    train_loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])

    # Фигура для графиков
    plt.figure(figsize=(10, 4))

    # 1) График LOSS
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label="Train Loss")
    plt.plot(val_loss, label="Val Loss")
    plt.title("Loss over epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # 2) Попробуем найти вторую метрику
    train_metric = history.history.get(metric, [])
    val_metric = history.history.get(f"val_{metric}", [])

    if len(train_metric) == 0 or len(val_metric) == 0:
        # Если метрика не найдена, выводим предупреждение
        plt.subplot(1, 2, 2)
        plt.text(0.5, 0.5, f"Metric '{metric}' not found in history!", 
                 ha='center', va='center', fontsize=12, color='red')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.title(f"Metric '{metric}' not in model history")
        plt.axis('off')
    else:
        # Строим кривые заданной метрики
        plt.subplot(1, 2, 2)
        plt.plot(train_metric, label=f"Train {metric}")
        plt.plot(val_metric, label=f"Val {metric}")
        plt.title(f"{metric.capitalize()} over epochs")
        plt.xlabel("Epoch")
        plt.ylabel(metric.capitalize())
        plt.legend()

    plt.tight_layout()
    plt.show()