# predict.py
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import cv2
import os

labels = ['star']


# Load the 28x28 sketch
img = Image.open("air_drawing.png").convert("L").resize((28, 28))
img = np.array(img).astype("float32") / 255.0
img = img.reshape(1, 28, 28, 1)

# Load the trained model
model = load_model("airdraw_model.keras", compile=False)

prediction = model.predict(img)
index = np.argmax(prediction)
confidence = float(np.max(prediction))

pred_label = labels[index]
print(f"Predicted: {pred_label} (confidence: {confidence:.2f})")

# Show corresponding real-world image
img_path = os.path.join("real_images", f"{pred_label}.jpg")
if os.path.exists(img_path):
    real_img = cv2.imread(img_path)
    cv2.imshow(f"Real Result: {pred_label}", real_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print(f"⚠️ Real image not found at {img_path}")
