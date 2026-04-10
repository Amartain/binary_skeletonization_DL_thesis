# Stage 1
from PIL import Image as im
from torch.utils.data import DataSet, random_split, DataLoader
from torchvision import transforms
import numpy as np

# ## Data Pipeline / Data preparation

# Questions before starting:
# - What 2 datasets to use? - SkeView: Kimia 99, Kimia 216
# - What libraries?
#     - PyTorch
#         - torch.utils.data import Dataset, Subset, random_split, DataLoader
#         - TorchVision = from torchvision import transforms
#     - NumPy
#     - plt - matplotlib # showing images in a grid and stuff  [1]
#     - PIL - Image [2]



# ### Access
# - access data in a reliable way, ensuring minimal RAM usage too - using lazy loading.
# #### DataSet class
# `__init__`: filepath, where to load from 
# `__len__` : gives back total samples 
# `__getitem__` : actually retrieving images - 1 at a time! 

# ### Quality - Data Transformations

# - This is where we do the transformations needed.
# #### Basic needed transformations
#     - ToTensor()
#     --- WHAT SIZE WOULD BE THE MOST appropiate???
#     - ReSize()  --- shortest edge only
#     - CenterCrop() --- the rest
#     - do we need normalization? for binary images no I don't think. its already bw 0-1

# #### TEST EVERY STEP!

# ### Efficiency
# #### Data Loader
# - ensuring train/val/test dataset split...
# - on the fly augmentation and its set up well --- making subset/wrapper class for the training data vs validation and testing one!











def SCH_single_image_show():
    img = im.open("data/from_SkeView/Kimia99\Kimia99-Original\Kimia99-Original/bonefishes.jpg")

    print(img.format, img.size, img.mode)

    img.show()

