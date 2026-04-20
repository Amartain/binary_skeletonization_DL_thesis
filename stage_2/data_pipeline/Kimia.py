from torch.utils.data import Dataset
import os
from PIL import Image
from pipeline import clean_labels


class Kimia(Dataset):
    """
    original_dir - path to directory with the original shapes - in jpg format - 
    gt_dir - path to directory with the ground truth "labels" / images - in png format -
    thumbs_dir - path to directory with the ground truths put onto to og shapes called "thumbs"
    - in png format -
    """
   
    def __init__(self, original_dir, gt_dir, thumbs_dir, transform=None):
        self.original_dir = original_dir
        self.gt_dir = gt_dir
        self.thumbs_dir = thumbs_dir
        self.transform = transform
    
        #TODO: make a function to fix label names that takes in a list and removes .jpg extension part fromm the end
        self.labels = clean_labels(os.listdir(original_dir))

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx): 
        label = self.labels[idx]

        original, gt, thumb, label = self.retrieve_image(label)

        if self.transform is not None:
            
            original = self.transform(original)
            gt = self.transform(gt)
            thumb = self.transform(thumb)


        return original, gt, thumb, label
    
    def retrieve_image(self, label): # use label cause we have diff. names for each image!
        """
        returns 3 items  the original and ground truth images and seperately the thumbs image
        unfort. the og. images are jpgs whilst the other 2 are pngs so we need sep...
        """
        #print(label)
        jpg_filename = label + ".jpg"
        png_filename = label + ".png"
        original_path = os.path.join(self.original_dir, jpg_filename)
        gt_path = os.path.join(self.gt_dir, png_filename)
        thumb_path = os.path.join(self.thumbs_dir, png_filename)

        # set mode to binary!!!
        original = Image.open(original_path).convert(mode="1")
        gt = Image.open(gt_path).convert(mode="1")
        thumb = Image.open(thumb_path).convert(mode="1")

        return original, gt, thumb, label
    
    def get_len(self):
        return len(self.labels)

    def get_labels(self):
        return self.labels