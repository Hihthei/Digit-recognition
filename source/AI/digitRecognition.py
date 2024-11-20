from pathlib import Path
import cv2 # computer vision (load images)
import numpy as np # numpy.array()
import matplotlib.pyplot as plt # visualation of the digit
import tensorflow as tf # machine learning


mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

models_dir = Path('AI/models')


def _model_name(model_path: Path) -> str:
    if model_path is None:
        model_path = Path("digitRecognition.keras")

    models_name = models_dir / model_path
    i = 1

    while Path(models_name).exists():
        models_name = models_dir / f"{model_path.stem}_{i}.keras"
        i += 1

    return str(models_name)


def train_model(train_session: int, model_path: Path):
    global x_train, x_test
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)


    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),

        tf.keras.layers.Conv2D(16, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(63, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.fit(
        x_train, y_train,
        epochs=train_session,
        validation_split=0.2,
        callbacks=[early_stopping]
    )

    model.save(_model_name(model_path))


def _load_model(model_path: Path):
    if not model_path:
        model_path = Path("digitRecognition.keras")

    models_name = models_dir / model_path

    try:
        print(f"Loading model: {models_name}")
        return tf.keras.models.load_model(models_name)
    except Exception as e:
        print("No valid model")
        print("Error:", e)
        return None


def evaluate_model(model_path: Path):
    model = _load_model(model_path)

    if model is None:
        print("Model is None.")
        return -1, -1

    loss, accuracy = model.evaluate(x_test, y_test)

    pourcent_loss = round(loss * 100, 2)
    pourcent_accuracy = round(accuracy * 100, 2)

    return pourcent_loss, pourcent_accuracy


def _display(img):
    plt.imshow(img[0], cmap=plt.cm.binary)
    plt.show()


def _image_normalize(image_path):
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (28, 28))
    img = np.invert(img)
    img = img / 255.0
    img = img.reshape(1, 28, 28)
    return img

'''    img = cv2.imread(str(image_path))[:,:,0]
    # img = cv2.resize(img, (28, 28))
    img = np.invert(np.array([img]))
    # img_for_display = img
    img = img.reshape(1, 28, 28)
    return img # img_for_model, img_for_display'''


def predict(image_path: Path, model_path: Path, model = None):
    if model is None:
        model = _load_model(model_path)
        if model is None:
            print("Model is None.")
            return None

    if not image_path.exists() or not image_path.suffix == '.png':
        print("Image is invalid.")
        return None

    try:
        # img_for_model, img = _image_normalize(image_path)
        img = _image_normalize(image_path)
        prediction = model.predict(img)
        confidence = prediction.max()
    except Exception as e:
        print("Error:", e)
        return None

    else:
        _display(img)
        return np.argmax(prediction), confidence


def test_model(model_path: Path):
    model = _load_model(model_path)

    image_dir = Path("AI/test_data")

    images = [f for f in image_dir.iterdir() if f.is_file() and f.suffix == '.png']

    results = []

    for image in images:
        result = -1
        try:
            result, confidence = predict(image, model_path, model)

            if result is None:
                print(f"Fail for image: {image}")
            else:
                print(f"This is probably a {result}!\n"
                      f"I'm sure at {confidence*100:.2f}%")

        except Exception as e:
            print("Error:", e)
        finally:
            answer = int(input("Correct answer: "))
            results.append((image.stem, result, confidence, answer))

    return results
