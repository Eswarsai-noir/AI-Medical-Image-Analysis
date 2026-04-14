import cv2
import numpy as np
import matplotlib.pyplot as plt

def predict_image(model, image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128,128))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)

    label = "PNEUMONIA" if pred > 0.5 else "NORMAL"
    return label, img

def show_prediction(model, image_path):
    label, img = predict_image(model, image_path)

    img = img[0]
    plt.imshow(img)
    plt.title(f"Prediction: {label}")
    plt.axis("off")
    plt.show()