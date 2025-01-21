from pathlib import Path
import cv2 # computer vision (load images)
import numpy as np # numpy.array()
import matplotlib.pyplot as plt # visualation of the digit
import tensorflow as tf # machine learning


class DigitRecognition:
    def __init__(self, model_name: str):
        self.__models_dir = Path('AI/models')
        self.__model_name = ""
        self.__model = None
        self.__history = None

        self.__x_train, self.__x_test, self.__y_train, self.__y_test = self.__load_and_preprocess_data()

        if self.__check_model(model_name):
            self.__model_name = model_name
            self.__load_model()


    def __load_and_preprocess_data(self):
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        x_train = tf.keras.utils.normalize(x_train, axis=1)
        x_test = tf.keras.utils.normalize(x_test, axis=1)

        x_train = np.expand_dims(x_train, axis=-1)
        x_test = np.expand_dims(x_test, axis=-1)

        return x_train, x_test, y_train, y_test


    def __get_model_path(self, model_name: str) -> Path:
        __model__path = self.__models_dir / model_name
        return __model__path


    def train_model(self, train_session: int):
        x_train = np.expand_dims(self.__x_train, axis=-1)
        y_train = self.__y_train

        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(28, 28, 1)),

            tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

            tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        __history = model.fit(
            x_train, y_train,
            epochs=train_session,
            validation_split=0.2,
            callbacks=[early_stopping]
        )

        model.save(self.__get_model_path(self.__model_name))
        self.__model = model
        self.__history = __history.history

        self.__display_training_history()


    def __check_model(self, model_name: str):
        __model_path = self.__get_model_path(model_name)

        if __model_path.exists():
            return True
        else:
            self.__model_name = "DigitRecognition.keras"
            return False


    def __load_model(self):
        try:
            variable = self.__get_model_path(self.__model_name)
            self.__model = tf.keras.models.load_model(variable)

            print(self.__model)

        except Exception as e:
            self.__model = None
            raise e


    def evaluate_model(self):
        if self.__model is None:
            raise ValueError("Can't evaluate model, model is not loaded")

        loss, accuracy = self.__model.evaluate(self.__x_test, self.__y_test)

        pourcent_loss = round(loss * 100, 2)
        pourcent_accuracy = round(accuracy * 100, 2)

        return pourcent_loss, pourcent_accuracy


    def __display(self, img):
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()


    def __display_training_history(self):
        plt.figure(figsize=(8, 6))

        # Perte
        plt.subplot(2, 1, 1)
        plt.plot(self.__history['loss'], label='Loss', marker='o')
        plt.plot(self.__history['val_loss'], label='Validation Loss', marker='x')
        plt.title('Loss Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        # Précision
        plt.subplot(2, 1, 2)
        plt.plot(self.__history['accuracy'], label='Accuracy', marker='o')
        plt.plot(self.__history['val_accuracy'], label='Validation Accuracy', marker='x')
        plt.title('Accuracy Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()


    def __image_normalize(self, image_path):
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (28, 28))
        img = np.invert(np.array(img))
        img = img / 255.0
        img = img.reshape(1, 28, 28)

        return img


    def predict(self, image_path: Path):
        if self.__model is None:
            raise ValueError("Can't evaluate model, model is not loaded")

        if not image_path.exists() or not image_path.suffix == '.png':
            raise ValueError("Image path is not valid")

        try:
            # img_for_model, img = _image_normalize(image_path)
            img = self.__image_normalize(image_path)
            prediction = self.__model.predict(img)
            confidence = prediction.max()
        except Exception as e:
            raise e

        else:
            self.__display(img)
            return np.argmax(prediction), confidence


    def test_model(self):
        if self.__model is None:
            raise ValueError("Can't evaluate model, model is not loaded")

        image_dir = Path("AI/test_data")

        images = [f for f in image_dir.iterdir() if f.is_file() and f.suffix == '.png']

        results = []
        count = 0

        for image in images:
            try:
                result, confidence = self.predict(image)

                if result is None:
                    print(f"Fail for image: {image}")
                else:
                    print(f"This is probably a {result}!\n"
                          f"I'm sure at {confidence * 100:.2f}%")

            except Exception as e:
                raise e
            else:
                answer = int(input("Correct answer: "))
                if result == answer:
                    count += 1

                results.append((image.stem, result, confidence, answer))

        accuracy = count/len(results)

        return results, accuracy
