import tensorflow as tf
import matplotlib.pyplot as plt

def load_data():
    """Загружает и возвращает данные CIFAR-10."""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    return (x_train, y_train, x_test, y_test), classes

def normalize_data(x_train, x_test):
    """Нормализует данные."""
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    return x_train, x_test

def prepare_labels(y_train, y_test, num_classes):
    """Преобразует метки в формат one-hot encoding."""
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)
    return y_train, y_test

def plot_sample_images(x_train, y_train, classes):
    """Показывает примеры изображений."""
    plt.figure(figsize=(10, 10))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(x_train[i])
        plt.title(classes[y_train[i][0]])
        plt.axis('off')
    plt.show()
