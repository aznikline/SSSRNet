from skimage.io import imread, imsave
from skimage import img_as_float
from skimage.transform import rotate, rescale
import sys
sys.path.append('..')
from os.path import join
import os
import random
import torch
from skimage.exposure import is_low_contrast
import torchvision.transforms as tf
#need do crop operation in Matlab

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

def load_data(root_dir, image_dir):
    #root_dir = "/tmp5/hang_data/T91_BSD200"
    image_set = join(root_dir, image_dir)

    images = [img_as_float(imread(join(image_set, x))) for x in sorted(os.listdir(image_set)) if is_image_file(x)]

    return images

def crop_random(img,cropx,cropy):
    y,x = img.shape[0:2]
    x1 = random.randint(0, x - cropx)
    y1 = random.randint(0, y - cropy)
    return img[y1:y1+cropy,x1:x1+cropx,:]


# Generator
def batch_generator(X, batch_size, scalar_scale=4 , isTest=False):
    #Test set generator
    if isTest == True:
        print("Test generate")
        targets = X["targets"]
        inputs = X["inputs"]
        #Y = [imresize(X[index], scalar_scale= 1/scalar_scale) for index in range(0, len(X))]
        #for index in range(0, len(X)):
            #HR = torch.cat((tf.ToTensor()(X[index]).unsqueeze(0),HR),0)
            #LR = torch.cat((tf.ToTensor()(Y[index]).unsqueeze(0),LR),0)
        batch_i = 0
        while batch_i < len(inputs):
            #yield  tf.ToTensor()(targets[batch_i]).unsqueeze(0), tf.ToTensor()(inputs[batch_i]).unsqueeze(0)
            yield torch.from_numpy(targets[batch_i].transpose((2, 0, 1))).unsqueeze(0).type(torch.FloatTensor), torch.from_numpy(inputs[batch_i].transpose((2, 0, 1))).unsqueeze(0).type(torch.FloatTensor)
            batch_i += 1

