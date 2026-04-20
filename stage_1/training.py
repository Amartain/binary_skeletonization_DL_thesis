from torch import nn, optim
from torch.utils.data import DataLoader
import torch
from simple_u_net_model import Simple_UNet
from data_pipeline import get_train_test_val_loaders
from stage1_tests import get_single_image_tensor_from_loader, test_show_image
from segmentation_models_pytorch import losses

TEST_MODE = True

# Setup
RANDOM_SEED = 42
generator = torch.manual_seed(RANDOM_SEED)
ALPHA = 0.1
BETA = 0.9 # 0.8 worked last time...

print(ALPHA,"* BCE", BETA, "* DICE")

# device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Cuda") if device == "cuda" else print("Training on CPU")

# Data
BATCH_SIZE = 32
KIMIA99 = 1
KIMIA216 = 2
# Model setup - see in the model code!

# Training Setup 
NO_EPOCHS = 1000
LR_RATE = 1e-4 # setup for Adam!
WEIGHT_DECAY = 1e-2 # set to 0 for stage 1!




def training_epoch(model, device, loss_function, optimizer, train_loader):
    epoch_loss = 0
    Dice_loss, BCE_loss = loss_function

    # prev. epoch ends w/ .eval() mode because of the validation epoch so reset
    model.train()

    running_loss = 0.0

    # I return thumbs and labels too but I actually don't care about it during training!
     # later TODO: - make monitoring by labels see which label is accessed how many times! - maybe graph distributions?
    y_distr = []

    for originals, skeletons, *_ in train_loader:
        # data setup
        x = originals.to(device)
        y = skeletons.to(device)
       # print("x min max", x.min(), x.max())
      #  print("y min max", y.min(), y.max())
        # 1.s vs. 0.s
        *_, val_counts = y.unique(return_counts=True) 
        
        count_0, count_1 = val_counts[0].item() / len(x), val_counts[1].item() / len(x)
       # print("VALCOUNTS: 0., 1.", count_0, count_1)
        distr = count_1 / count_0
        y_distr.append(distr)

        # forward pass
        optimizer.zero_grad()
      #  print("X shape", x.size())

        output = model(x)
      #  print("OUTPUT SHAPE: ", output.size())

        # backward pass
        BCE_loss_ = ALPHA * BCE_loss(output, y)
        Dice_loss_ = BETA * Dice_loss(output,y)
        loss = BCE_loss_ + Dice_loss_

      #  print("loss SHAPE: ", loss.size())


        loss.backward()

        optimizer.step()

        # Accumulative LOSS
        running_loss += loss.item() * len(x) # avarage out w/ batch_size to ensure same weight for all

    epoch_loss = running_loss / len(train_loader.dataset)
    #print("Y distribution: ", y_distr)

    return epoch_loss


def val_epoch(model, device, loss_function, val_loader):
    running_loss = 0.0

    # set model to eval!
    model.eval()

    # no need for grad.
    with torch.no_grad():
        # I will only care about thumbs and labels when doing visual analysis...
        for originals, skeletons, *_ in val_loader:
            x = originals.to(device)
            y = skeletons.to(device)

            output = model(x)

            BCE_loss_ = ALPHA * BCE_loss(output, y)
            Dice_loss_ = BETA * Dice_loss(output,y)
            loss = BCE_loss_ + Dice_loss_


            running_loss += loss.item() * len(x)


    epoch_val_loss = running_loss / len(val_loader.dataset)

    return epoch_val_loss

def train_model(model, device,  optimizer, train_loader, val_loader, loss_function):
    train_losses = []
    val_losses = []
    for epoch in range(NO_EPOCHS):
        print("STARTING Epoch ", epoch, "/", NO_EPOCHS)

        epoch_train_loss = training_epoch(model, device, loss_function, optimizer, train_loader)
        epoch_val_loss = val_epoch(model,device, loss_function, val_loader)

        print("Training Loss: ", epoch_train_loss)
        print("Validation Loss: ", epoch_val_loss)
        print("-"*80)

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
    

    print("Training finished")
    # I wanna plot this later
    return model, train_losses, val_losses

def visualize_results(model, val_loader):
    image, skeleton, thumbs, label = next(iter(val_loader))

   # move input to device the model dwells on
    image = image.to(device)

    model.eval()
    pred = model(image)

    print("Prediction mean", pred.mean(), "Prediction median", pred.median(), "highest pred. ", pred.max(), "Pred min.", pred.min())
    print("Unique prediction", pred.unique())
    # transitioning it to 1s and zeroes by masking
    pred_mask = pred < 0.5 # because I wanna have white for skel. black for background like og. image
    # converting to [0., 1.] image!
    pred = pred_mask.to(torch.float32)
    
    print(pred.unique())


    # moving back to CPU for visualization & detaching from grads.
    image = image.to("cpu")
    
    pred = pred.detach()
    print("after detach()", pred.unique())
    pred = pred.to("cpu")
    print("after 2 cpu", pred.unique())

    # select an image & sqeeze down so batch dimension is gone!
    image = image[0].squeeze()
    pred = pred[0].squeeze()

    print("after pred[0].squeeze()", pred.size(), pred.unique())

    test_show_image(image)
    test_show_image(pred)

# TRAINING Setup
# TODO: model.to(device) implementation instead!!!!!!

print("0. INIT MODEL")
model = Simple_UNet()
model.to(device)

print("1. Setting up Loss & optims")
BCE_loss = nn.BCELoss() # this isn't ideal / good because of HUGE class imbalance but this was just first step to test!
# smooth  aka. like gamma in Focal loss to account even more for class imbalance
Dice_loss = losses.DiceLoss('binary', from_logits=False)


# for some reason Adam didn't really seem to work - no learning vs... 0.09 both on val and training dataset!
optimizer = optim.AdamW(model.parameters(), lr=LR_RATE, weight_decay=WEIGHT_DECAY) 
# optimizer = optim.SGD(model.parameters(), lr=LR_RATE, weight_decay=WEIGHT_DECAY)

# data
print("2. Setting up DATALOADERs")
train_loader, val_loader, test_loader = get_train_test_val_loaders(KIMIA99)

print("3. STARTING TRAINING")
print("_"*80)
trained_model, train_losses, val_losses = train_model(model, device, optimizer, train_loader, val_loader, loss_function=(Dice_loss, BCE_loss))

print("VISUALIZE RESULTS")
print("-"*80)
visualize_results(trained_model, val_loader)




