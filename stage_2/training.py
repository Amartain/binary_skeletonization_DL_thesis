from torch import nn, optim
from torch.utils.data import DataLoader
import torch
from torch.utils.tensorboard import SummaryWriter
from models.UNet import UNet
from data_pipeline.pipeline import get_train_test_val_loaders
from segmentation_models_pytorch import losses, metrics
from models.ResidualAttentionUNet import Residual_Attention_UNet
import torchvision


print(torch.__version__)

import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version PyTorch was built with: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"Current device: {torch.cuda.get_device_name(0)}")
else:
    import os
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

TEST_MODE = True

# Setup
RANDOM_SEED = 42
generator = torch.manual_seed(RANDOM_SEED)
ALPHA = 0.3
BETA = 0.7 # 0.8 worked last time... - (0.2-4, 0.7-0.8) works well higher and model looses confidence + starts breaking lines
# alpha, beta chosen 0.3 and 0.7 weights for simplicity
abs = [(0.3, 0.7)] #

print(ALPHA,"* BCE", BETA, "* DICE")

# device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Cuda") if device == "cuda" else print(device)

# Data
BATCH_SIZE = 32
KIMIA99 = 1
KIMIA216 = 2
batch_sizes = [16]
# Model setup - see in the model code!

# Training Setup 
NO_EPOCHS = 300
LR_RATE = 1e-4 # setup for Adam!
WEIGHT_DECAY = 1e-2 # set to default for baseline

# TensorBoard
step = 0
FULL_PATH_TO_LOGDIR = f"runs\\UNet\\Kimia99\\ComboLoss_Baseline_300EPOCHS"

def training_epoch(model, device, loss_function, optimizer, train_loader, alpha=ALPHA, beta=BETA):
    epoch_loss = 0
    
    if type(loss_function) == tuple:
        Dice_loss, BCE_loss = loss_function

    # prev. epoch ends w/ .eval() mode because of the validation epoch so reset
    model.train()

    running_loss = 0.0

    # I return thumbs and labels too but I actually don't care about it during training!
     # later TODO: - make monitoring by labels see which label is accessed how many times! - maybe graph distributions?
    y_distr = []
    running_f1 = 0.0
    
    # used for passing up first batch for visualization!
    batch_count = 0
    first_batch = None

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


        # backward pass
        if type(loss_function) == tuple:
            BCE_loss_ = alpha * BCE_loss(output, y)
            Dice_loss_ = beta * Dice_loss(output,y)
            loss = BCE_loss_ + Dice_loss_
        else: loss = loss_function(output, y)

      # metrics & for tensorboard
        if batch_count == 0:
            first_batch = (
               originals.cpu(),
               skeletons.cpu(),
               output.detach().cpu()
            )

        # calculating f1 score for the batch
        target = y.round().long()
        tp, fp, fn, tn = metrics.get_stats(output, target, mode='binary', threshold=0.5)
        f1 = metrics.f1_score(tp, fp, fn, tn, reduction='micro').item()
        running_f1 += f1 * len(x)
        
        loss.backward()

        optimizer.step()

        # Accumulative LOSS
        running_loss += loss.item() * len(x) # avarage out w/ batch_size to ensure same weight for all

        batch_count += 1

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_f1 = running_f1 / len(train_loader.dataset)

    return epoch_loss, epoch_f1, first_batch

def val_epoch(model, device, loss_function, val_loader, alpha=ALPHA, beta=BETA):
    running_loss = 0.0
    running_f1 = 0.0
    batch_count = 0
    # set model to eval!
    model.eval()


    if type(loss_function) == tuple:
        BCE_loss, Dice_loss = loss_function

    # no need for grad.
    with torch.no_grad():
        # I will only care about thumbs and labels when doing visual analysis...
        for originals, skeletons, *_ in val_loader:
            x = originals.to(device)
            y = skeletons.to(device)

            output = model(x)

            if type(loss_function) == tuple:
                BCE_loss_ = alpha * BCE_loss(output, y)
                Dice_loss_ = beta * Dice_loss(output,y)
                loss = BCE_loss_ + Dice_loss_
            else: loss = loss_function(output, y)

                  # metrics & for tensorboard
            if batch_count == 0:
                first_batch = (
                originals.cpu(),
                skeletons.cpu(),
                output.detach().cpu()
                )

            # calculating batch f1 score
            target = y.round().long()
            tp, fp, fn, tn = metrics.get_stats(output, target, mode='binary', threshold=0.5)
            f1 = metrics.f1_score(tp, fp, fn, tn, reduction='micro').item()
            running_f1 += f1 * len(x)


            running_loss += loss.item() * len(x)


    epoch_val_loss = running_loss / len(val_loader.dataset)
    epoch_f1 = running_f1 / len(val_loader.dataset)


    return epoch_val_loss, epoch_f1, first_batch

def make_image_grid(image_batch):
    """
    image_batch - tuple (original image, ground_truth_image, model_output) ON CPU already!
    """
    imgs, gt_imgs, ypreds = image_batch
    
    cat_images = torch.cat((imgs, gt_imgs, ypreds), dim=3)
    
    return torchvision.utils.make_grid(cat_images, nrow=1)
    


def train_model(model, device,  optimizer, train_loader, val_loader, loss_function, alpha=ALPHA, beta=BETA, batch_size=BATCH_SIZE):
    train_losses = []
    val_losses = []
    step = 0

    tb_writer = SummaryWriter(f"{FULL_PATH_TO_LOGDIR}\\{batch_size}ALPHA{alpha}BETA{beta}")

    image, *_ = next(iter(train_loader))
    image = image.to(device)

    tb_writer.add_graph(model, image)

    for epoch in range(NO_EPOCHS):
        

        epoch_train_loss, epoch_train_f1_acc, train_1st_batch = training_epoch(model, device, loss_function, optimizer, train_loader, alpha, beta)
        epoch_val_loss, epoch_val_f1_acc, val_1st_batch = val_epoch(model,device, loss_function, val_loader, alpha, beta)

        if epoch % 25 == 0 or (epoch+1) == NO_EPOCHS:
            print("Epoch ", epoch, "/", NO_EPOCHS)
            print("Training Loss: ", epoch_train_loss)
            print("Validation Loss: ", epoch_val_loss)
            print("/"*80)
            print("Training Accuracy (F1)", epoch_train_f1_acc)
            print("Validation Accuracy (F1)", epoch_val_f1_acc)
            print("_"*80)

            # visual stuff to tensorboard
            tb_writer.add_image("Training", make_image_grid(train_1st_batch), global_step=step)
            tb_writer.add_image("Validation", make_image_grid(val_1st_batch), global_step=step)
            

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)

        # TensorBoard
        tb_writer.add_scalars("Train vs Validation Loss", {"Training Loss": epoch_train_loss, "Validation Loss": epoch_val_loss}, global_step=step)
        tb_writer.add_scalars("Training vs Validation ACCURACY (F1)", {"Training Acc":epoch_train_f1_acc, "Validation Accuracy":epoch_val_f1_acc}, global_step=step)
        tb_writer.add_hparams({"alpha":alpha, "beta":beta, "batch_size":batch_size}, {"Training loss":epoch_train_loss, "Validation Loss":epoch_val_loss, "Training Accuracy F1":epoch_train_f1_acc,"Validation Accuracy F1":epoch_val_f1_acc}, global_step=step)

        


        step += 1
    

    print("Training finished")
    # I wanna plot this later
    return model, train_losses, val_losses

# TRAINING Setup
# TODO: model.to(device) implementation instead!!!!!!
def train(batch_size=BATCH_SIZE, lr=LR_RATE, weight_decay = WEIGHT_DECAY):
    for alpha, beta in abs:
            print("0. INIT MODEL")
            torch.manual_seed(RANDOM_SEED) #need for reproducibility - weight init.
            model = UNet()
            #model = Residual_Attention_UNet(residual=False, attention=True)
            model.to(device)

            print("1. Setting up Loss & optims")
            BCE_loss = nn.BCELoss() # this isn't ideal / good because of HUGE class imbalance but this was just first step to test!
            # smooth  aka. like gamma in Focal loss to account even more for class imbalance
            Dice_loss = losses.DiceLoss('binary', from_logits=False)


            # for some reason Adam didn't really seem to work - no learning vs... 0.09 both on val and training dataset!
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay) 
            # optimizer = optim.SGD(model.parameters(), lr=LR_RATE, weight_decay=WEIGHT_DECAY)

            # data
            print("2. Setting up DATALOADERs")
            train_loader, val_loader, test_loader = get_train_test_val_loaders(KIMIA99, batch_size)

            print("3. STARTING TRAINING")
            print("_"*80)
            trained_model, train_losses, val_losses = train_model(model, device, optimizer, train_loader, val_loader, loss_function=(BCE_loss, Dice_loss), alpha=alpha, beta=beta, batch_size=batch_size)

            print("VISUALIZE RESULTS")
            print("-"*80)
            #visualize_results(trained_model, val_loader)

            
t = 0

for batch_size in batch_sizes:
    print("TRAIN no.", t)
    train(batch_size)
    t += 1

