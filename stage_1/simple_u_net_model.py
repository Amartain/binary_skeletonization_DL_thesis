from torch import manual_seed, nn, cat
from data_pipeline import get_train_test_val_loaders
import numpy as np
from stage1_tests import show_image_from_tensor


# Setup
RANDOM_SEED = 42
GENERATOR = manual_seed(RANDOM_SEED)
BATCH_SIZE = 16
STARTING_FEATURE_NO = 16
TEST_MODE = False

# MODEL SETUP
STRIDE = 2
POOL_TRANSPOSE_KERNEL_SIZE = (2,2)
KERNEL_SIZE = (3,3)
PADDING = "same"
OUT_CHANNELS = 16 # doubled w/ every down!


device = None

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
    
 def __init__(self, input_channels=1, out_channels=OUT_CHANNELS, kernel_size=KERNEL_SIZE, padding=PADDING, device=None):
        super().__init__()
    
        self.conv_block = nn.Sequential(
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=out_channels,
                kernel_size=kernel_size, 
                padding=padding,
                device=device
                ),
            nn.ReLU()      
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size, 
                padding=padding,
                device=device
                ),
            nn.ReLU()   
        )


 def forward(self, x):
        x = self.conv_block(x)
        x = self.conv_block2(x)
        return x



class EncoderBlock(nn.Module):
    def __init__(self, input_channels=1, out_channels=OUT_CHANNELS, kernel_size=KERNEL_SIZE, padding=PADDING, device=None):
        super().__init__()
        
        self.down_block = CNNBlock(input_channels,out_channels, kernel_size, padding, device)
        self.maxpool = nn.MaxPool2d(kernel_size=POOL_TRANSPOSE_KERNEL_SIZE, stride=STRIDE, device=device)

    def forward(self, x):
        feature_map = self.down_block(x)
        x = self.maxpool(feature_map)

        return  x, feature_map


# - Bottleneck / Bridge - no Pool
#     - so regular ass convolution w/o pool - so just use CNNBlock class



class Bridge(nn.Module):
    def __init__(self, input_channels, out_channels, kernel_size=KERNEL_SIZE, padding=PADDING, device=None):
         super().__init__()

         self.conv_block = CNNBlock(input_channels, out_channels, kernel_size, padding, device)
         self.up_conv = nn.ConvTranspose2d(input_channels, out_channels, padding, kernel_size=POOL_TRANSPOSE_KERNEL_SIZE, stride=STRIDE, device=device)
         
         

    def forward(self, x):
        x = self.conv_block(x)
        x = self.up_conv(x)

        return x
# - Decoder - Upsampling - Mode for skeletons gotta be: nearest neighbour not bilinear!???
#     - UPSAMPLE (convtranspose)  & 1/2x channels  >> CONV BLOCK >> UPSAMPLE >> next conv block...
#     - Upsampling via: ConvTranspose 

class DecoderBlock(nn.Module):
    def __init__(self, input_channels, out_channels, kernel_size=KERNEL_SIZE, padding=PADDING, device=None):
        super().__init__()

        self.conv_block = CNNBlock(input_channels, out_channels, kernel_size, padding, device)
        self.up_conv = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=POOL_TRANSPOSE_KERNEL_SIZE, stride=STRIDE)

# - Connecting paths
#      - concatanation that's it just cat... meow
#         - cat places convoluted image at that stage ALONGSIDE the decoded features!

    def forward(self, x, feature_map):
        x = cat(feature_map,x,dim=1)
        x = self.conv_block(x)
        x = self.up_conv(x)

        return x

class simple_UNet(nn.Module):
    def __init__(self, input_channels, kernel_size=KERNEL_SIZE, padding=PADDING, device=None):
        super().__init__()

        # Encoder / Down 
        # defaul start in: 1 then 16 > 32 > 64 > 128
        # DOUBLE out_channels every block!
        self.encoder_block1 = EncoderBlock(input_channels,OUT_CHANNELS)
        self.encoder_block2 = EncoderBlock(input_channels=OUT_CHANNELS, out_channels=OUT_CHANNELS*2)
        self.encoder_block3 = EncoderBlock(input_channels=OUT_CHANNELS*2, out_channels=OUT_CHANNELS*4)
        self.encoder_block4 = EncoderBlock(input_channels=OUT_CHANNELS*4, out_channels=OUT_CHANNELS*8)

        # Bridge
        # in channels default - 128 > 256
        self.bridge = Bridge(input_channels=OUT_CHANNELS*8, out_channels=OUT_CHANNELS*16)

        # TODO: fix decoder channels - w/ cat half etc think through!
        # Decoder / Up
        # start channel default 256 > 128 > 64 > 32 > 16
        # HALF start channel every block
        self.decoder_block1 = DecoderBlock(input_channels=OUT_CHANNELS*16, out_channels=OUT_CHANNELS*8)
        self.decoder_block2 = DecoderBlock(input_channels=OUT_CHANNELS*8, out_channels=OUT_CHANNELS*4)
        self.decoder_block3 = DecoderBlock(input_channels=OUT_CHANNELS*4, out_channels=OUT_CHANNELS*2)
        self.decoder_block4 = DecoderBlock(input_channels=OUT_CHANNELS*2, out_channels=OUT_CHANNELS)

        # Final Prediction layer
        # - OUT: convolution final time w/o w/ SIGMOID for binary image segmentation
        self.prediction = nn.Sequential(
            nn.Conv2d(in_channels=OUT_CHANNELS, out_channels=1, kernel_size=(1,1),padding=PADDING, device=device),
            nn.Sigmoid()
        )

# let PyTorch __call__ handle the magic apperantly 
    def forward(self, x):

        # Encoder / Down
        x, feature_map_1 = self.encoder_block1(x)
        x, feature_map_2 = self.encoder_block2(x)
        x, feature_map_3 = self.encoder_block3(x)
        x, feature_map_4 = self.encoder_block4(x)

        # Bridge
        x = self.bridge(x)

        # Decoder / Up
        x = self.decoder_block1(x, feature_map_4)
        x = self.decoder_block2(x, feature_map_3)
        x = self.decoder_block3(x, feature_map_2)
        x = self.decoder_block4(x, feature_map_1)

        # final prediction w/ sigmoid
        x = self.prediction(x)

        return x





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
        
        print("TESTING Xs")
        detached_x = x[0].squeeze().detach()
        show_image_from_tensor(detached_x)

        logs.append("testing FORWARD method")
        output = model.forward(x)

        

        logs.append("MODEL OUTPUT")
        if (isinstance(output, tuple)):
            conv_x, x = output
            logs.append("conv_x Size: ")
            logs.append(conv_x.size())
            logs.append("x Size: ")
            logs.append(x.size())
            
            show_image_from_tensor(conv_x[0,1].detach())

            show_image_from_tensor(x[0,1].detach())
        else:
            logs.append(output.size())
            show_image_from_tensor(output[0,1].detach())
    except Exception as e:
        logs.append("ERORR!")
        logs.append(e)
    
    return logs 

if TEST_MODE:
    model = EncoderBlock()
    dataset = KIMIA216
    print(test_model_class(model, dataset))
