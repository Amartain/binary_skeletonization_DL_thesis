from PIL import Image
import matplotlib.pyplot as plt
from torch import tensor, zeros
import numpy as np



# test vars
# tens = zeros(128,128)
filler_lines = "_"*80

# old stuff will delete and refactor:
test_mode = True
print_mode = True

def show_image(img):
    if img is not None: img.show()


def tensor_to_image(img_tensor_on_CPU):
    logs = []
    title = "STARTING TENSOR TO NUMPY then SHOW...."
    print(title)
    logs.append(title)
    try:
        print("HAHOO")
        img_as_np = np.array(img_tensor_on_CPU, dtype=np.uint8)
        logs.append("TRYING TO SHOW IMAGE")
        logs.append(filler_lines)
        plt.imshow(img_as_np, cmap="Grays")
        plt.show()
        logs.append(list(np.unique(img_as_np, return_counts=True)))

    except Exception as e:
        logs.append(["Upps ERROR: ", e])
        return None, logs

    return logs

def show_image_from_tensor(img_tensor_on_CPU):
    logs = []
    logs.append(tensor_to_image(img_tensor_on_CPU))

    print(logs)

    return logs 


# MODEL TESTS

# Test: model parameters how many total?
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
            
            show_image_from_tensor(conv_x[0,0].detach())

            show_image_from_tensor(x[0,0].detach())
        else:
            logs.append(output.size())
            show_image_from_tensor(output[0,0].detach())
    except Exception as e:
        logs.append("ERORR!")
        logs.append(e)
    
    return logs 


## TEST SECTION ##
#  TEST EVERY STEP!
# dataset to be tested
# dataset = Kimia(kimia99_original_dir, kimia99_gt_dir, kimia99_thumb_dir, transform)

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
    print(image_tensor.unique(return_counts=True, sorted=True)) 

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




