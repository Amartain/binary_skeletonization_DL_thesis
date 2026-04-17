from torch import nn, optim
from torch.utils.data import DataLoader
import torch
from simple_u_net_model import Simple_UNet
from data_pipeline import get_train_test_val_loaders
from stage1_tests import get_single_image_tensor_from_loader, test_show_image

TEST_MODE = True

# Setup
RANDOM_SEED = 42
generator = torch.manual_seed(RANDOM_SEED)

# device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Cuda") if device == "cuda" else print("Training on CPU")

# Data
BATCH_SIZE = 16
KIMIA99 = 1
KIMIA216 = 2
# Model setup - see in the model code!

# Training Setup 
NO_EPOCHS = 50
LR_RATE = 0.01 # setup for Adam!
WEIGHT_DECAY = 0.1 # set to 0 for stage 1!




def training_epoch(model, device, loss_function, optimizer, train_loader):
    epoch_loss = 0

    # prev. epoch ends w/ .eval() mode because of the validation epoch so reset
    model.train()

    running_loss = 0.0

    # I return thumbs and labels too but I actually don't care about it during training!
     # later TODO: - make monitoring by labels see which label is accessed how many times! - maybe graph distributions?

    for originals, skeletons, *_ in train_loader:
        # data setup
        x = originals.to(device)
        y = skeletons.to(device)

        # forward pass
        optimizer.zero_grad()

        output = model.forward(x)

        # backward pass
        loss = loss_function(output, y)

        loss.backward()

        optimizer.step()

        # Accumulative LOSS
        running_loss += loss.item() * len(x) # avarage out w/ batch_size to ensure same weight for all

    epoch_loss = running_loss / len(train_loader.dataset)

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

            loss = loss_function(output,y)

            running_loss += loss.item() * len(x)


    epoch_val_loss = running_loss / len(val_loader.dataset)

    return epoch_val_loss

def train_model(model, device, loss_function, optimizer, train_loader, val_loader):
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
    pred_mask = pred > 0.5
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
model = Simple_UNet(device=device)

print("1. Setting up Loss & optims")
BCE_loss = nn.BCELoss() # this isn't ideal / good because of HUGE class imbalance but this was just first step to test!

# for some reason Adam didn't really seem to work - no learning vs... 0.09 both on val and training dataset!
# # optimizer = optim.Adam(model.parameters(), lr=LR_RATE, weight_decay=WEIGHT_DECAY) 
optimizer = optim.SGD(model.parameters(), lr=LR_RATE, weight_decay=WEIGHT_DECAY)

# data
print("2. Setting up DATALOADERs")
train_loader, val_loader, test_loader = get_train_test_val_loaders(KIMIA99)

print("3. STARTING TRAINING")
print("_"*80)
trained_model, train_losses, val_losses = train_model(model, device, BCE_loss, optimizer, train_loader, val_loader)

print("VISUALIZE RESULTS")
print("-"*80)
visualize_results(trained_model, val_loader)




