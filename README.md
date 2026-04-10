# binary_skeletonization_DL_thesis

Libraries used: PyTorch

# Stage 1

## Data Pipeline / Data preparation

Questions before starting:
- What 2 datasets to use?
- What libraries?
    - PyTorch
        - torch.utils.data import Dataset, Subset, random_split, DataLoader
        - TorchVision = from torchvision import transforms
    - NumPy
    - plt - matplotlib # showing images ???? [1]
    - PIL - Image [2]



### Access
- access data in a reliable way, ensuring minimal RAM usage too - using lazy loading.
#### DataSet class
`__init__`: filepath, where to load from 
`__len__` : gives back total samples 
`__getitem__` : actually retrieving images - 1 at a time! 

### Quality - Data Transformations

- This is where we do the transformations needed.
#### Basic needed transformations
    - ToTensor()
    --- WHAT SIZE WOULD BE THE MOST appropiate???
    - ReSize()  --- shortest edge only
    - CenterCrop() --- the rest
    - do we need normalization? for binary images no I don't think. its already bw 0-1

#### TEST EVERY STEP!

### Efficiency
#### Data Loader
- ensuring train/val/test dataset split...
- on the fly augmentation and its set up well --- making subset/wrapper class for the training data vs validation and testing one!

## Model Architecture

## Training Epoch

## Validation Epoch

## Result Visualization


[1] https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html
[2] https://pillow.readthedocs.io/en/stable/handbook/tutorial.html


# Stage 3
### Data augmentation
    - vertical, horizontal flip (mirroring)
    - rotations: 90, 180, 270
