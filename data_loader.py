import os
import cv2
import numpy as np

def load_data(data_dir, img_size=224):
    images = []
    labels = []

    categories = ["NORMAL", "PNEUMONIA"]

    for label, category in enumerate(categories):
        path = os.path.join(data_dir, category)

        for img in os.listdir(path):
            try:
                img_path = os.path.join(path, img)
                image = cv2.imread(img_path)
                image = cv2.resize(image, (128, 128))

                images.append(image)
                labels.append(label)
            except:
                continue

    return np.array(images, dtype=np.float32), np.array(labels)