# Stage 1
from PIL import Image
from torch.utils.data import Dataset, random_split, DataLoader
from torchvision import transforms
from torchvision.transforms import v2
import os
from torch import manual_seed, float32

# Setup parameters
RANDOM_SEED = 42
GENERATOR = manual_seed(RANDOM_SEED) # for reproducability leave it at that ! 
IMAGE_SIZE = 160
# commit msg.

# ## Data Pipeline / Data preparation

# Path variables
# TODO: ask myself if I need these as constants or normal vars.

kimia99_original_dir = r"data\kimia99_dataset\Kimia99-Original"
kimia99_gt_dir = r"data\kimia99_dataset\Kimia99-GT"
kimia99_thumb_dir = r"data\kimia99_dataset\Kimia99-Thumb"

kimia216_original_dir = r"data\kimia216_dataset\Kimia216-Original"
kimia216_gt_dir = r"data\kimia216_dataset\Kimia216-GT"
kimia216_thumb_dir = r"data\kimia216_dataset\Kimia216-Thumb"

# TODO: Make a function to visualize images - (loaded, transformed, outputs)
# TODO: refactor old test code to use log [] lists instead of prints and only print 1x / stage! 
# (started data laoding and stuff like that - THAT WE WANT TO SEE IMMEDIATELY!!!1

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


class Kimia(Dataset):
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

        original, gt, thumb, label = self.retrieve_image(label)

        if self.transform is not None:
            
            original = self.transform(original)
            gt = self.transform(gt)
            thumb = self.transform(thumb)
        #print(idx)

        return original, gt, thumb, label
    
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

        # set mode to binary!!!
        original = Image.open(original_path).convert(mode="1")
        gt = Image.open(gt_path).convert(mode="1")
        thumb = Image.open(thumb_path).convert(mode="1")

        return original, gt, thumb, label
    
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
        transforms.CenterCrop(IMAGE_SIZE),
        v2.Compose([v2.ToImage(), v2.ToDtype(float32, scale=False)]) # binary images don't need scaling!
    ]
)

# ### Efficiency
# #### Data Loader
# - ensuring train/val/test dataset split...
# - on the fly augmentation and its set up well 
# --- making subset/wrapper class for the training data vs validation and testing one!

def train_val_test_split(dataset, test_friction=0.15, val_friction=0.15):
    """
    IN: Dataset class, friction sizes of test and validation datasets
    OUT: read test, validation and train datasets!
    """
  

    val_len = int(len(dataset) * val_friction)
    test_len = int(len(dataset) * test_friction)
    train_len = len(dataset) - val_len - test_len
    

    train_dataset, val_dataset, test_dataset = random_split(dataset=dataset, lengths=[train_len, val_len, test_len], generator=GENERATOR)

    return train_dataset, val_dataset, test_dataset


# DataSet Numbers
# 1 = Kimia99
# 2 = Kimia216

# Putting together so we have model input
def get_train_test_val_loaders(dataset_no, batch_size):
    print("Getting loaders", "."*70)
    if dataset_no == 1: # Kimia99
        dataset = Kimia(kimia99_original_dir,kimia99_gt_dir, kimia99_thumb_dir, transform=transform)
    elif dataset_no == 2:
        dataset = Kimia(kimia216_original_dir, kimia216_gt_dir, kimia216_thumb_dir, transform=transform)
    else:
        print("Dataset Number doesn't exist.")
        return None    

    train_dataset, val_dataset, test_dataset = train_val_test_split(dataset=dataset)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=GENERATOR)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) 
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  

    print("DataLoaders ready", "."*70)

    return train_loader, val_loader, test_loader


