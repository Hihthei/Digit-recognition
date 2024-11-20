from AI.digitRecognition import train_model, evaluate_model, test_model, predict

from pathlib import Path

image_name = Path("AI/test_data/0_image_4.png")

model_path = Path("test.keras")



def main_predict():
    prediction, confidence = predict(image_name, model_path, None)

    if prediction:
        print(f"This is probably a {prediction}!\n"
              f"I'm sure at {confidence * 100:.2f}%")

def main_train():
    train_model(5, model_path)

def main_evaluate():
    loss, accuracy = evaluate_model(model_path)
    print(f"Model performance: Loss-> {loss}%, Accuracy-> {accuracy}%")

def main_test():
    results = test_model(model_path)

    count = 0

    for result in results:
        image_nb, prediction, confidence, answer = result
        print(  f"For image: {image_nb}\n"
                f"Prediction: {prediction}\n"
                f"Confidence: {confidence}\n"
                f"Correct answer: {answer}\n")
        if prediction == answer:
            count += 1

    print(f"Accuracy: {count/len(results)}")


main_predict()
