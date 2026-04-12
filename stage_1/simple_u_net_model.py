from torch import manual_seed, nn


# Setup
RANDOM_SEED = 42
GENERATOR = manual_seed(random_seed)
BATCH_SIZE = 16
STARTING_FEATURE_NO = 16


# ## Model Architecture [9]
# - Encoder - Downsampling
#     - CONV BLOCK >> DOWNSAMPLE (maxpool) & 2x channels >> next conv block >> down...
#     - saves convoluted output BEFORE maxpool every time because we will do skip connections with it !
#     - 2x CHANNELS after each conv block 
#     - PADDING: make so it remains the SAME SIZE! - saves headache
class ConvBlock(nn.Module):
    
 def __init__(self, input_channels=1, output_channels=16, kernel_size=(3,3), padding='same'):
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

 def forward(self, x):
        x = self.conv_block(x)
        return x

class CNNBlock(nn.Module):
    def __init__(self, input_channels=1, output_channels_1st=16, kernel_size=(3,3), padding='same'):
        self.in_channels = input_channels
        self.out_channels_1st = output_channels_1st
        self.kernel_size = kernel_size
        self.padding = padding

        self.conv_block = ConvBlock(self.in_channels, self.out_channels_1st, self.kernel_size, self.padding)
        self.conv_block2 = ConvBlock(self.out_channels_1st, self.out_channels_1st * 2, self.kernel_size, self.padding)
                

    def forward(self, x):
        x = self.conv_block(x)
        x = self.conv_block2(x)
        return x



class EncoderBlock(nn.Module):
    def __init__(self, input_channels=1, output_channels_1st=16, kernel_size=(3,3), padding='same'):
        self.in_channels = input_channels
        self.out_channels_1st = output_channels_1st
        self.kernel_size = kernel_size
        self.padding = padding
        
        self.encoder_block = CNNBlock( in_channels=self.in_channels,out_channels=self.out_channels_1st, kernel_size=self.kernel_size, padding=self.padding)
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
