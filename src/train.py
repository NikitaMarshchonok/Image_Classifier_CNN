import os
from src.data_preparation import load_data, normalize_data
from src.model_builder import build_model

def save_training_plots(history):
    """Сохраняет графики обучения."""
    import os
    import matplotlib.pyplot as plt
    os.makedirs('plots', exist_ok=True)

    # График точности
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.savefig('plots/accuracy.png')
    plt.show()

    # График потерь
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig('plots/loss.png')
    plt.show()




def train_model():
    """Загружает данные, строит модель и обучает ее."""
    (x_train, y_train, x_test, y_test), classes = load_data()
    x_train, x_test = normalize_data(x_train, x_test)

    model = build_model()
    history = model.fit(x_train, y_train, epochs=14, validation_data=(x_test, y_test))

    # Сохранение модели
    os.makedirs('models', exist_ok=True)
    model.save('models/cifar10_model.keras')
    print("Model saved to 'models/cifar10_model.keras'")
    return history


if __name__ == "__main__":
    train_model()
