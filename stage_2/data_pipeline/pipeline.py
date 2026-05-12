# Stage 1
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
from torchvision.transforms import v2
from torch import manual_seed, float32
from data_pipeline.datasets.Kimia import Kimia

# Setup parameters
RANDOM_SEED = 42
GENERATOR = manual_seed(RANDOM_SEED) # for reproducability leave it at that ! 
IMAGE_SIZE = 160


# ## Data Pipeline / Data preparation

# Path variables
# TODO: ask myself if I need these as constants or normal vars.

kimia99_original_dir = r"data\kimia99_dataset\Kimia99-Original"
kimia99_gt_dir = r"data\kimia99_dataset\Kimia99-GT"
kimia99_thumb_dir = r"data\kimia99_dataset\Kimia99-Thumb"

kimia216_original_dir = r"data\kimia216_dataset\Kimia216-Original"
kimia216_gt_dir = r"data\kimia216_dataset\Kimia216-GT"
kimia216_thumb_dir = r"data\kimia216_dataset\Kimia216-Thumb"


# Helper functions


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


