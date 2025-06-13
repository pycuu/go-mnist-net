from PIL import Image
import numpy as np
import sys
import json

def preprocess_image(image_path):
    image = Image.open(image_path).convert("L")  # grayscale
    image = image.resize((28, 28))
    data = np.asarray(image).astype("float32")
    data = 1-(data / 255.0)  # MNIST format
    return data.flatten().tolist()

if __name__ == "__main__":
    path = sys.argv[1]
    input_vector = preprocess_image(path)
    print(json.dumps(input_vector))
