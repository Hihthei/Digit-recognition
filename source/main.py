from AI.digitRecognition import DigitRecognition

from pathlib import Path
import os

image_path = Path("AI/test_data")
image_name = Path("0_image_0.png")


if __name__ == "__main__":
    DigitModel = DigitRecognition("DigitRecognition.keras")

    status = "continue"

    while status == "continue":
        os.system("clear")
        print("Selected image: ", image_name, "\n",
            "1- Select image\n",
            "2- Train model\n",
            "3- Evaluate model\n",
            "4- Test model\n",
            "5- Predict\n"
        )
        break
