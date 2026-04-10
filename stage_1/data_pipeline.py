# Stage 1
from PIL import Image
from torch.utils.data import Dataset, random_split, DataLoader
from torchvision import transforms
import numpy as np
import os

# Setup parameters
test_mode = True
batch_size = 16

# ## Data Pipeline / Data preparation

# Path variables

kimia99_original_dir = r"data\kimia99_dataset\Kimia99-Original"
kimia99_gt_dir = r"data\kimia99_dataset\Kimia99-GT"
kimia99_thumb_dir = r"data\kimia99_dataset\Kimia99-Thumb"


# Helper functions

def clean_labels(jpg_filenames):
    labels = [filename.replace(".jpg", "") for filename in jpg_filenames]

    return labels


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
        self.original_dir = original_dir
        self.gt_dir = gt_dir
        self.thumbs_dir = thumbs_dir
        self.transform = transform
        
        #TODO: make a function to fix label names that takes in a list and removes .jpg extension part fromm the end
        self.labels = clean_labels(os.listdir(original_dir))

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx): 
        label = self.labels[idx]

        original, gt, thumb = self.retrieve_image(label)

        if self.transform is not None:
            
            original = self.transform(original)
            gt = self.transform(gt)
            thumb = self.transform(thumb)
        #print(idx)

        return original, gt, thumb
    
    def retrieve_image(self, label): # use label cause we have diff. names for each image!
        """
        returns 3 items  the original and ground truth images and seperately the thumbs image
        unfort. the og. images are jpgs whilst the other 2 are pngs so we need sep...
        """
        #print(label)
        jpg_filename = label + ".jpg"
        png_filename = label + ".png"
        original_path = os.path.join(self.original_dir, jpg_filename)
        gt_path = os.path.join(self.gt_dir, png_filename)
        thumb_path = os.path.join(self.thumbs_dir, png_filename)

        original = Image.open(original_path).convert(mode="L")
        gt = Image.open(gt_path).convert(mode="L")
        thumb = Image.open(thumb_path).convert(mode="L")

        return original, gt, thumb
    
    def get_len(self):
        return len(self.labels)

    def get_labels(self):
        return self.labels




# ### Quality - Data Transformations

# - This is where we do the transformations needed.
# #### Basic needed transformations
#     - ToTensor()
#     --- WHAT SIZE WOULD BE THE MOST appropiate???
#     - ReSize()  --- shortest edge only
#     - CenterCrop() --- the rest
#     - do we need normalization? for binary images no I don't think. its already bw 0-1

# Transforms + Data Loader
# simple tensor conversion for now because we in stage 1 we don't do data augmentation!
transform = transforms.Compose(
    [
        transforms.Resize(120), #ensuring 120 min
        transforms.CenterCrop(120),
        transforms.ToTensor()
    ]
)

kimia99 = Kimia99(kimia99_original_dir, kimia99_gt_dir, kimia99_thumb_dir, transform)

# single batch for test
single_batch_loader = DataLoader(kimia99, batch_size=batch_size, shuffle=False)

idx = 0
for originals, gts, thumbs in single_batch_loader:
    print("sUCESS BATCH: ", idx, "-"*32)
    idx += 1

    

# #### TEST EVERY STEP!

# ### Efficiency
# #### Data Loader
# - ensuring train/val/test dataset split...
# - on the fly augmentation and its set up well 
# --- making subset/wrapper class for the training data vs validation and testing one!

## TEST SECTION ##

def test_single_image_show(filepath):
    #img2show = None
    e = None
    try:
        with Image.open(filepath) as img:
                print(img.format, img.size, img.mode)
                img.show()
    except e:
        print(e)

def test_dataset_kimia99():


    kimia99 = Kimia99(kimia99_original_dir, kimia99_gt_dir, kimia99_thumb_dir)
    # print(kimia99.get_len())
    # print(kimia99.get_labels())


### RUNNING ALL TESTS
def tests(test_mode):
    if(test_mode):
        try:
            # poor bonefishes will be used forever lol
            filepath = kimia99_original_dir + r"\bonefishes.jpg"

            # test_single_image_show(filepath)
            test_dataset_kimia99()
            print("SUCCESSFUL TESTS")
        except:
            print("TESTS FAILED")


    else:
        pass

tests(test_mode)