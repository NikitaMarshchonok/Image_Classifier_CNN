import tensorflow as tf
from src.data_preparation import load_data, normalize_data, prepare_labels

def evaluate_model(model_path='models/cifar10_model.keras'):
    """Оценивает модель на тестовых данных и визуализирует результаты."""
    # Загрузка данных
    (x_train, y_train, x_test, y_test), classes = load_data()
    x_train, x_test = normalize_data(x_train, x_test)

    # Преобразование меток в формат one-hot encoding
    y_train, y_test = prepare_labels(y_train, y_test, len(classes))

    # Загрузка модели
    model = tf.keras.models.load_model(model_path)

    # Компиляция модели (на случай, если она не была скомпилирована)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Оценка модели
    loss, accuracy = model.evaluate(x_test, y_test, verbose=2)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print(f"Test Loss: {loss:.4f}")

    # Предсказания на первых 9 изображениях
    predictions = model.predict(x_test[:9])
    for i, prediction in enumerate(predictions[:5]):
        print(f"Image {i + 1}: Predicted: {classes[prediction.argmax()]}, Actual: {classes[y_test[i].argmax()]}")

    # Визуализация предсказаний
    plot_predictions(x_test[:9], y_test[:9], predictions, classes)


def plot_predictions(x_test, y_test, predictions, classes):
    """Показывает изображения с предсказаниями."""
    import matplotlib.pyplot as plt
    import numpy as np

    plt.figure(figsize=(10, 10))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(x_test[i])
        pred_label = classes[np.argmax(predictions[i])]  # Предсказанный класс
        true_label = classes[np.argmax(y_test[i])]  # Истинный класс
        plt.title(f"Pred: {pred_label}\nTrue: {true_label}", fontsize=10)
        plt.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Укажите путь к модели, если он отличается
    evaluate_model(model_path='models/cifar10_model.keras')
