# Stage 1
from PIL import Image as im
from torch.utils.data import Dataset, random_split, DataLoader
from torchvision import transforms
import numpy as np
import os

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

kimia99_original_dir = "data/kimia99_dataset/Kimia99-Original"
kimia99_gt_dir = "data/kimia99_dataset/Kimia99-GT"
kimia99_thumb_dir = "data/kimia99_dataset/Kimia99-Thumb"


# ### Access
# - access data in a reliable way, ensuring minimal RAM usage too - using lazy loading.
# #### DataSet class
# `__init__`: filepath, where to load from 
# `__len__` : gives back total samples 
# `__getitem__` : actually retrieving images - 1 at a time! 


class Kimia99(Dataset):
    """
    original_dir - path to directory with the original shapes - in jpg format - 
    gt_dir - path to directory with the ground truth "labels" / images - in png format -
    thumbs_dir - path to directory with the ground truths put onto to og shapes called "thumbs"
    - in png format -
    """
   
    def __init__(self, original_dir, gt_dir, thumbs_dir, transform=None):
        self.root_original_dir = original_dir
        self.root_gt_dir = gt_dir
        self.root_thumbs_dir = thumbs_dir
        self.transform = transform
        
        self.shapes_dir = os.path.join(self.root_original_dir, "jpg")
        self.gt_dir = os.path.join(self.root_gt_dir, "png")
        self.root_thumbs_dir = os.path.join(self.root_thumbs_dir, "png")

        print(self.shapes_dir)

    def __len__(self):
        return 99



kimia99 = Kimia99(kimia99_original_dir, kimia99_gt_dir, kimia99_thumb_dir)


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






## TEST SECTION ##


def test_single_image_show(filepath):
    #img2show = None
    e = None
    try:
        with im.open(filepath) as img:
                print(img.format, img.size, img.mode)
                img.show()
    except e:
        print(e)


### RUNNING ALL TESTS
def tests():
    # poor bonefishes will be used forever lol
    filepath = kimia99_original_dir + "/bonefishes.jpg"
    print(filepath)

    test_single_image_show(filepath)