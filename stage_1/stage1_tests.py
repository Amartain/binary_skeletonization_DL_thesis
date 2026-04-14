from PIL import Image
import matplotlib.pyplot as plt
from torch import tensor, zeros
import numpy as np


# test vars
# tens = zeros(128,128)
filler_lines = "_"*80

def show_image(img):
    if img is not None: img.show()


def tensor_to_image(img_tensor_on_CPU):
    logs = []
    title = "STARTING TENSOR TO NUMPY then SHOW...."
    print(title)
    logs.append(title)
    try:
        print("HAHOO")
        img_as_np = np.array(img_tensor_on_CPU, dtype=np.uint8)
        logs.append("TRYING TO SHOW IMAGE")
        logs.append(filler_lines)
        plt.imshow(img_as_np, cmap="Grays")
        plt.show()
        logs.append(list(np.unique(img_as_np, return_counts=True)))

    except Exception as e:
        logs.append(["Upps ERROR: ", e])
        return None, logs

    return logs

def show_image_from_tensor(img_tensor_on_CPU):
    logs = []
    logs.append(tensor_to_image(img_tensor_on_CPU))

    print(logs)

    return logs 

    