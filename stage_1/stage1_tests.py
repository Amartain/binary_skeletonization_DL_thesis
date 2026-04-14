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
        img_as_np = np.array(img_tensor_on_CPU, dtype=np.uint8) * 255 # scaling up!
        logs.append(img_as_np)
        logs.append(np.shape(img_as_np))
        print(logs)
        logs.append("TRYING TO SHOW IMAGE")
        logs.append(filler_lines)
        img = Image.fromarray(img_as_np,mode="1") # 1 binary???
    except Exception as e:
        logs.append(["Upps ERROR: ", e])
        return None, logs

    return img, logs

def show_image_from_tensor(img_tensor_on_CPU):
    logs = []
    img, logs_ = tensor_to_image(img_tensor_on_CPU)
    logs.append(logs_)
    print(logs)
    if img_tensor_on_CPU is not None:
        try:
            show_image(img)
        except Exception as e:
            logs.append(e)
            return logs
    else:
        logs.append("show_image_from_tensor(img_tensor_on_CPU): Recieved None Image")
        return logs

    return logs 

    