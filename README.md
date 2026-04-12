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
#### DataSet class [4] [5]
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
#### Data Loader [3]
- ensuring train/val/test dataset split...
- on the fly augmentation and its set up well --- making subset/wrapper class for the training data vs validation and testing one!
- using [6] [7] MANUAL seed for reproducibility AND to ensure test set ALWAYS remains a test set! (only used once I finished stage one and stopped wanting to improve!)
- ensuring fixed train/val/test -- and stupid subsets [8]

## Model Architecture [9]
- Encoder - Downsampling
    - CONV BLOCK >> DOWNSAMPLE (maxpool) & 2x channels >> next conv block >> down...
    - saves convoluted output BEFORE maxpool every time because we will do skip connections with it !
    - 2x CHANNELS after each conv block 
    - PADDING: make so it remains the SAME SIZE! - saves headache [10]
- Bottleneck / Bridge - no Pool
    - so regular ass convolution w/o pool
- Decoder - Upsampling - Mode for skeletons gotta be: nearest neighbour not bilinear!
    - UPSAMPLE (convtranspose)  & 1/2x channels  >> CONV BLOCK >> UPSAMPLE >> next conv block...
    - Upsampling via: ConvTranspose 
- Connecting paths
     - concatanation that's it just cat... meow
        - cat places convoluted image at that stage ALONGSIDE the decoded features!
- OUT: convolution final time w/o w/ SIGMOID for binary image segmentation

Test: model parameters how many total?

MODEL improvement for stage 2/3/4: different initializiation states - he_normal etc
- different U-Net models: Residual U-Net, Attention U-Net


## Training Epoch

## Validation Epoch

## Result Visualization

# Resources
[1] https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html
[2] https://pillow.readthedocs.io/en/stable/handbook/tutorial.html
[3] https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html
[4] https://www.geeksforgeeks.org/python/how-to-convert-images-to-numpy-array/
[5] https://docs.pytorch.org/docs/stable/generated/torch.from_numpy.html
[6] https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.random_split
[7] https://docs.pytorch.org/docs/stable/generated/torch.Generator.html#torch.Generator
[8] https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.Subset
[9] https://www.codegenes.net/blog/unet-segmentation-pytorch/
[10] https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html


# Stage 3
### Data augmentation
    - vertical, horizontal flip (mirroring)
    - rotations: 90, 180, 270
