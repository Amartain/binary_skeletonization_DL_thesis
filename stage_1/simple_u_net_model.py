from torch import manual_seed, nn

from data_pipeline import get_train_test_val_loaders


# Setup
RANDOM_SEED = 42
GENERATOR = manual_seed(RANDOM_SEED)
BATCH_SIZE = 16
STARTING_FEATURE_NO = 16
test_mode = True

# Dataset Numbers
KIMIA99 = 1
KIMIA216 = 2


# ## Model Architecture [9]
# - Encoder - Downsampling
#     - CONV BLOCK >> DOWNSAMPLE (maxpool) & 2x channels >> next conv block >> down...
#     - saves convoluted output BEFORE maxpool every time because we will do skip connections with it !
#     - 2x CHANNELS after each conv block 
#     - PADDING: make so it remains the SAME SIZE! - saves headache
class CNNBlock(nn.Module):
    
 def __init__(self, input_channels=1, output_channels=16, kernel_size=(3,3), padding='same'):
        super().__init__()
        self.in_channels = input_channels
        self.out_channels = output_channels
        self.kernel_size = kernel_size
        self.padding = padding
    
        self.conv_block = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size, 
                padding=self.padding
                ),
            nn.ReLU()      
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.out_channels,
                out_channels=self.out_channels * 2,
                kernel_size=self.kernel_size, 
                padding=self.padding
                ),
            nn.ReLU()      
        )


 def forward(self, x):
        x = self.conv_block(x)
        x = self.conv_block2(x)
        return x



class EncoderBlock(nn.Module):
    def __init__(self, input_channels=1, output_channels_1st=16, kernel_size=(3,3), padding='same'):
        super().__init__()
        self.in_channels = input_channels
        self.out_channels_1st = output_channels_1st
        self.kernel_size = kernel_size
        self.padding = padding
        
        self.encoder_block = CNNBlock(self.in_channels,self.out_channels_1st, self.kernel_size, self.padding)
        self.maxpool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))

    def forward(self, x):
        conv_x = self.encoder_block(x)
        x = self.maxpool(conv_x)

        return conv_x, x


# - Bottleneck / Bridge - no Pool
#     - so regular ass convolution w/o pool
# - Decoder - Upsampling - Mode for skeletons gotta be: nearest neighbour not bilinear!
#     - UPSAMPLE (convtranspose)  & 1/2x channels  >> CONV BLOCK >> UPSAMPLE >> next conv block...
#     - Upsampling via: ConvTranspose 
# - Connecting paths
#      - concatanation that's it just cat... meow
#         - cat places convoluted image at that stage ALONGSIDE the decoded features!
# - OUT: convolution final time w/o w/ SIGMOID for binary image segmentation

# Test: model parameters how many total?

# MODEL improvement for stage 2/3/4: different initializiation states - he_normal etc
# - different U-Net models: Residual U-Net, Attention U-Net


# TESTS 

# For testing variables done

# Path variables

def test_model_class(model, dataset_no):
    print("started class test")
    logs = []
    
    try: 
        logs.append("MODEL INFO")


        logs.append(str(model))
        
        logs.append("DATA LOADER run")

        train_loader, val_loader, test_loader = get_train_test_val_loaders(dataset_no)

        # getting 1 item from the train loader just the original image the rest 
        # skeleton (y), thumbs and labels we don't care about right now
        x, *_ = next(iter(train_loader)) # * puts all return arg there into a list _ is ignorable namign conv
        logs.append(x.size())
        
        logs.append("testing FORWARD method")
        output = model.forward(x)

        logs.append("MODEL OUTPUT")
        if (isinstance(output, tuple)):
            conv_x, x = output
            logs.append("conv_x Size: ")
            logs.append(conv_x.size())
            logs.append("x Size: ")
            logs.append(x.size())
        else:
            logs.append(output.size())
    except Exception as e:
        logs.append("ERORR!")
        logs.append(e)
    
    return logs 




if test_mode:
    model = EncoderBlock()
    dataset = KIMIA216
    print(test_model_class(model, dataset))
