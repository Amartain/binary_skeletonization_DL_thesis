# Stage 1
from PIL import Image
from torch.utils.data import Dataset, random_split, DataLoader
from torch import Generator, tensor
from torchvision import transforms
from torchvision.transforms import v2
import numpy as np
import os
from torch import manual_seed, float32
from stage1_tests import show_image_from_tensor

# Setup parameters
test_mode = False
print_mode = False
batch_size = 14
random_seed = 42
generator = manual_seed(random_seed) # for reproducability leave it at that ! 
IMAGE_SIZE = 160

# ## Data Pipeline / Data preparation

# Path variables

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
    
    generator = Generator().manual_seed(random_seed)

    train_dataset, val_dataset, test_dataset = random_split(dataset=dataset, lengths=[train_len, val_len, test_len], generator=generator)

    return train_dataset, val_dataset, test_dataset


# DataSet Numbers
# 1 = Kimia99
# 2 = Kimia216

# Putting together so we have model input
def get_train_test_val_loaders(dataset_no):
    print("Getting loaders", "."*70)
    if dataset_no == 1: # Kimia99
        dataset = Kimia(kimia99_original_dir,kimia99_gt_dir, kimia99_thumb_dir, transform=transform)
    elif dataset_no == 2:
        dataset = Kimia(kimia216_original_dir, kimia216_gt_dir, kimia216_thumb_dir, transform=transform)
    else:
        print("Dataset Number doesn't exist.")
        return None    

    train_dataset, val_dataset, test_dataset = train_val_test_split(dataset=dataset)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) 
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  

    print("DataLoaders ready", "."*70)

    return train_loader, val_loader, test_loader


## TEST SECTION ##
#  TEST EVERY STEP!
# dataset to be tested
# dataset = Kimia(kimia99_original_dir, kimia99_gt_dir, kimia99_thumb_dir, transform)
dataset = Kimia(kimia216_original_dir, kimia216_gt_dir, kimia216_thumb_dir, transform)

def get_single_image_tensor_from_loader(data_loader):

    img_tensor = next(iter(data_loader))[1][0] # original/skel/thumb image col+ label, getting 1 single sample!
    img_tensor = img_tensor.squeeze() # reducing (1,128,128) to just (128,128) cause that's what we need
    img_tensor = tensor(img_tensor)

    return img_tensor

### Comment out unused tests
def test_show_loader_outputs(dataset_no):
    train_loader, test_loader, val_loader = get_train_test_val_loaders(dataset_no)
    
    train_img = get_single_image_tensor_from_loader(train_loader)
    test_show_image(train_img)
    val_img = get_single_image_tensor_from_loader(val_loader)
    test_show_image(val_img)
    test_img = get_single_image_tensor_from_loader(test_loader)
    test_show_image(test_img)







def test_show_image(image_tensor):
    # ensuring we have an actual image
    # print(train_image_1.unique(return_counts=True, sorted=True)) 

    show_image_from_tensor(image_tensor)

def test_get_train_test_val_loaders(dataset_no):
    logs = []
    title = "TESTING DATASET: [" + str(dataset_no) + "]"
    logs.append(title)

   # TODO finish
    try:
        logs.append("Testing GET LOADERS if RUN")
        train_loader, val_loader, test_loader = get_train_test_val_loaders(dataset_no)

        logs.append("Testing TRAIN Loader object")
        logs.append(list(next(iter(train_loader))))
        logs.append("Testing TEST Loader object")
        logs.append(list(next(iter(test_loader))))
        logs.append("Testing VAL Loader object")
        logs.append(list(next(iter(test_loader))))

        

    except Exception as e:
        logs.append("ERROR!: ")
        logs.append(e)

    return logs 

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


def test_dataset_class(dataset):
    print(len(dataset))
    
    print("Testing DataLoader")
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
def tests(print_mode):
    if(test_mode):
        try:
            print('Testing |"Mother"| Dataset')
            dataset_labels = test_dataset_class(dataset)
            print('Testing Train/Val/Test Split')
            train_ds, val_ds, test_ds =  test_train_val_test_split(dataset)

            if print_mode:
                print("-"*80, "testing Test Dataset")
                test_labels = test_dataset_class(test_ds)
                print("-"*80, "testing Val Dataset")
                val_labels = test_dataset_class(val_ds)
                print("-"*80, "testing Train Dataset")
                train_labels = test_dataset_class(train_ds) 


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

def test_with_logs(test_mode):
    logs = []
    if test_mode:


        logs.append("KIMIA 99 DATALOADER TEST Started")
        kimia99_test = test_get_train_test_val_loaders(dataset_no=1)
        logs.append("KIMIA 216 DATALOADER TEST STARTED")
        kimia216_test = test_get_train_test_val_loaders(dataset_no=2)

    return logs




if print_mode and test_mode:
    print(test_get_train_test_val_loaders(1))
    print(test_get_train_test_val_loaders(2))
    # tests(print_mode=True)
elif test_mode:
    tests(print_mode=False)
    test_show_loader_outputs(1)

