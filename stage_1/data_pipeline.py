# Stage 1
from PIL import Image
from torch.utils.data import Dataset, random_split, DataLoader
from torch import Generator
from torchvision import transforms
import numpy as np
import os
from torch import manual_seed

# Setup parameters
test_mode = True
print_mode = False
batch_size = 14
random_seed = 42
manual_seed(random_seed) # for reproducability leave it at that ! 

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

        original = Image.open(original_path).convert(mode="L")
        gt = Image.open(gt_path).convert(mode="L")
        thumb = Image.open(thumb_path).convert(mode="L")

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
        transforms.Resize(120), #ensuring 120 min
        transforms.CenterCrop(120),
        transforms.ToTensor()
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
    
    generator = Generator().manual_seed(random_seed)

    train_dataset, val_dataset, test_dataset = random_split(dataset=dataset, lengths=[train_len, val_len, test_len], generator=generator)

    return train_dataset, val_dataset, test_dataset



## TEST SECTION ##
#  TEST EVERY STEP!
# dataset to be tested
dataset = Kimia99(kimia99_original_dir, kimia99_gt_dir, kimia99_thumb_dir, transform)



def test_train_val_test_split(dataset):
    val_len = 0.1
    test_len = 0.1

    train_dataset, val_dataset, test_dataset = train_val_test_split(dataset)

    print(len(train_dataset), len(val_dataset), len(test_dataset), sep="/" )

    return train_dataset, val_dataset, test_dataset
    

def test_single_image_show(filepath):
    #img2show = None
    e = None
    try:
        with Image.open(filepath) as img:
                print(img.format, img.size, img.mode)
                img.show()
    except Exception as e:
        print(e)


def test_dataset(dataset):

    # print(dataset.get_len())
    # print(dataset.get_labels())
    
    # single batch for test
    single_batch_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    idx = 0
    labels = []
    for originals, gts, thumbs, label in single_batch_loader:
        idx += 1
       # print(idx, label)
        labels.append(label)
    print("test_dataset run successfully")

    return labels
### RUNNING ALL TESTS
def tests(test_mode, print_mode):
    if(test_mode):
        try:
            # poor bonefishes will be used forever lol
            filepath = kimia99_original_dir + r"\bonefishes.jpg"

            # test_single_image_show(filepath)
            dataset_labels = test_dataset(dataset)
            train_ds, val_ds, test_ds =  test_train_val_test_split(dataset)

            if print_mode:
                print("-"*80, "testing Test Dataset")
                test_labels = test_dataset(test_ds)
                print("-"*80, "testing Val Dataset")
                val_labels = test_dataset(val_ds)
                print("-"*80, "testing Val Dataset")
                train_labels = test_dataset(train_ds) 


                print("="*80)
                print("Test Dataset Labels: ")
                print(test_labels)

                print("="*80)
                print("Val Dataset Labels: ")
                print(val_labels)

                print("="*80)
                print("Train Dataset Labels: ")
                print(train_labels)
            

            print("TESTS RUN")
        except Exception as e:
            print("TESTS FAILED reason: \n", e)


    else:
        pass

tests(test_mode, print_mode)