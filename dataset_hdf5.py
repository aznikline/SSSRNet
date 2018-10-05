import torch.utils.data as data
import torch
from skimage.io import imread, imsave
import torchvision.transforms as tf
import numpy as np
import random
import os
import glob
import h5py
from skimage.transform import rotate
from skimage import img_as_float
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt

class MergedDataset(data.Dataset):
    def __init__(self, path_patients):
        hdf5_list = [x for x in glob.glob(os.path.join(path_patients, '*.h5'))]  # only h5 files
        self.datasets = []
        self.datasets_gt = []
        self.total_count = 0
        self.limits = []
        for f in hdf5_list:
            h5_file = h5py.File(f, 'r+')
            dataset = h5_file['data']
            dataset_gt = h5_file['label']
            self.datasets.append(dataset)
            self.datasets_gt.append(dataset_gt)
            self.limits.append(self.total_count)
            self.total_count += len(dataset)
            # print 'len ',len(dataset)
            # print self.limits

    def __getitem__(self, index):

        dataset_index = -1
        # print 'index ',index
        for i in range(len(self.limits) - 1, -1, -1):
            # print 'i ',i
            if index >= self.limits[i]:
                dataset_index = i
                break
        # print 'dataset_index ',dataset_index
        assert dataset_index >= 0, 'negative chunk'

        in_dataset_index = index - self.limits[dataset_index]

        # Data Augment:
        LR_patch = self.datasets[dataset_index][in_dataset_index]
        HR_patch = self.datasets_gt[dataset_index][in_dataset_index]
        flip_channel = random.randint(0, 1)
        if flip_channel != 0:
            LR_patch = np.flip(LR_patch, 2)
            HR_patch = np.flip(HR_patch, 2)
        # randomly rotation
        rotation_degree = random.randint(0, 3)
        self.datasets[dataset_index][in_dataset_index] = np.rot90(LR_patch, rotation_degree, (1,2))
        self.datasets_gt[dataset_index][in_dataset_index] = np.rot90(HR_patch, rotation_degree, (1,2))
        # Save patches
        #LR = torch.from_numpy(self.datasets[dataset_index][in_dataset_index]).float()
        #HR = torch.from_numpy(self.datasets_gt[dataset_index][in_dataset_index]).float()
        #LR = tf.ToPILImage()(LR)
        #HR = tf.ToPILImage()(HR)
        #LR.save('result/LR_patch/{0:08d}.jpg'.format(index))
        #HR.save('result/HR_patch/{0:08d}.jpg'.format(index))
        #print(index)

        return torch.from_numpy(self.datasets[dataset_index][in_dataset_index]).float(), \
               torch.from_numpy(self.datasets_gt[dataset_index][in_dataset_index]).float()

    def __len__(self):
        return self.total_count


class DatasetFromHdf5_DP(data.Dataset):
    def __init__(self, file_path):
        super(DatasetFromHdf5_DP, self).__init__()
        hf = h5py.File(file_path)
        self.data = hf.get("data")
        self.target = hf.get("label")
        '''''''''
        for index in range(0, len(self.data)):
            LR = torch.from_numpy(self.data[index, :, :, :]).float()
            HR = torch.from_numpy(self.target[index, :, :, :]).float()
            LR = tf.ToPILImage()(LR)
            HR = tf.ToPILImage()(HR)
            LR.save('result/LR/{0:04d}.jpg'.format(index))
            HR.save('result/HR/{0:04d}.jpg'.format(index))
            print(index)
        '''''

    def __getitem__(self, index):
        # randomly flip
        #print(index)
        #data shppe: C*H*W
        LR_patch = self.data[index, :, :, :]
        HR_patch = self.target[index, :, :, :]
        LR_patch = LR_patch.transpose(0,2,1)
        HR_patch = HR_patch.transpose(0,2,1)
        LR_patch = np.asarray(LR_patch, np.float32)
        HR_patch = np.asarray(HR_patch, np.float32)
        return LR_patch.copy(), HR_patch.copy()


    def __len__(self):
        return self.data.shape[0]

class DatasetFromHdf5_SR(data.Dataset):
    def __init__(self, file_path):
        super(DatasetFromHdf5_SR, self).__init__()
        hf = h5py.File(file_path)
        self.data = hf.get("lr")
        self.target = hf.get("hr")
        '''''''''
        for index in range(0, len(self.data)):
            LR = torch.from_numpy(self.data[index, :, :, :]).float()
            HR = torch.from_numpy(self.target[index, :, :, :]).float()
            LR = tf.ToPILImage()(LR)
            HR = tf.ToPILImage()(HR)
            LR.save('result/LR/{0:04d}.jpg'.format(index))
            HR.save('result/HR/{0:04d}.jpg'.format(index))
            print(index)
        '''''
    def __getitem__(self, index):
        # randomly flip
        #print(index)
        #data shppe: C*H*W
        LR_patch = self.data[index, :, :, :]
        HR_patch = self.target[index, :, :, :]

        LR_patch = LR_patch.transpose(0,2,1)
        HR_patch = HR_patch.transpose(0,2,1)

        flip_channel = random.randint(0, 1)
        if flip_channel != 0:
            LR_patch = np.flip(LR_patch, axis=2)
            HR_patch = np.flip(HR_patch, axis=2)

        LR_patch = np.asarray(LR_patch, np.float32)
        HR_patch = np.asarray(HR_patch, np.float32)

        return LR_patch.copy(), HR_patch.copy()

    def __len__(self):
        return self.data.shape[0]

class DatasetFromHdf5_OLD(data.Dataset):
    def __init__(self, file_path):
        super(DatasetFromHdf5_OLD, self).__init__()
        hf = h5py.File(file_path)
        self.data = hf.get("lr")
        self.target = hf.get("hr")
        self.label = hf.get("label")
        is_Test = False

        if is_Test == True :
            for index in range(0,min(100,len(self.data))):
                LR = torch.from_numpy(self.data[index, :, :, :]).float()
                HR = torch.from_numpy(self.target[index, :, :, :]).float()
                LABEL = torch.from_numpy(self.label[index,:,:]).float()
                LR = tf.ToPILImage()(LR)
                HR = tf.ToPILImage()(HR)
                LABEL = tf.ToPILImage()(LABEL)
                LR.save('result/LR/{0:04d}.jpg'.format(index))
                HR.save('result/HR/{0:04d}.jpg'.format(index))
                LABEL.save('result/LABEL/{0:04d}.jpg'.format(index))
                print(index)

    def __getitem__(self, index):
        # randomly flip
        #print(index)
        #data shppe: C*H*W
        LR_patch = self.data[index, :, :, :]
        HR_patch = self.target[index, :, :, :]
        Label_patch = self.label[index, :, :]

        Label_patch = np.expand_dims(Label_patch, axis=0)
        Label_patch = np.asarray(Label_patch, np.float32)
        LR_patch = np.asarray(LR_patch, np.float32)
        HR_patch = np.asarray(HR_patch, np.float32)

        LR_patch = LR_patch.transpose((0,2,1))
        HR_patch = HR_patch.transpose((0,2,1))
        Label_patch = Label_patch.transpose((0,2,1))

        flip_channel = random.randint(0, 1)
        if flip_channel != 0:
            LR_patch = np.flip(LR_patch, axis=2)
            HR_patch = np.flip(HR_patch, axis=2)
            Label_patch = np.flip(Label_patch, axis=2)

        # randomly rotation
        #rotation_degree = random.randint(0, 3)
        #self.data[index, :, :, :] = np.rot90(LR_patch, rotation_degree, (1,2))
        #self.target[index, :, :, :] = np.rot90(HR_patch, rotation_degree, (1,2))
        return LR_patch.copy(), HR_patch.copy(), Label_patch.copy()

    def __len__(self):
        return self.data.shape[0]


class DatasetFromHdf5(data.Dataset):
    def __init__(self, file_path):
        super(DatasetFromHdf5, self).__init__()
        hf = h5py.File(file_path)
        self.data = hf.get("lr")
        self.target = hf.get("hr")
        self.label = hf.get("label")
        self.hr_label = hf.get("hr_label")
        is_Test = False

        if is_Test == True :
            for index in range(0,min(100,len(self.data))):
                LR = torch.from_numpy(self.data[index, :, :, :]).float()
                HR = torch.from_numpy(self.target[index, :, :, :]).float()
                LABEL = torch.from_numpy(self.label[index,:,:]).float()
                LR = tf.ToPILImage()(LR)
                HR = tf.ToPILImage()(HR)
                LABEL = tf.ToPILImage()(LABEL)
                LR.save('result/LR/{0:04d}.jpg'.format(index))
                HR.save('result/HR/{0:04d}.jpg'.format(index))
                LABEL.save('result/LABEL/{0:04d}.jpg'.format(index))
                print(index)

    def __getitem__(self, index):
        # randomly flip
        #print(index)
        #data shppe: C*H*W
        LR_patch = self.data[index, :, :, :]
        HR_patch = self.target[index, :, :, :]
        Label_patch = self.label[index, :, :]
        HR_Label_patch = self.hr_label[index, :, :]

        Label_patch = np.expand_dims(Label_patch, axis=0)    #升维
        HR_Label_patch = np.expand_dims(HR_Label_patch, axis=0)
        Label_patch = np.asarray(Label_patch, np.float32)
        HR_Label_patch = np.asarray(HR_Label_patch, np.float32)   #转float32
        LR_patch = np.asarray(LR_patch, np.float32)
        HR_patch = np.asarray(HR_patch, np.float32)

        LR_patch = LR_patch.transpose((0,2,1))    #matlab cxwxh 转 cxhxw
        HR_patch = HR_patch.transpose((0,2,1))
        Label_patch = Label_patch.transpose((0,2,1))
        HR_Label_patch = HR_Label_patch.transpose((0, 2, 1))

        flip_channel = random.randint(0, 1)    #翻转 随机数干啥的
        if flip_channel != 0:
            LR_patch = np.flip(LR_patch, axis=2)
            HR_patch = np.flip(HR_patch, axis=2)
            Label_patch = np.flip(Label_patch, axis=2)
            HR_Label_patch = np.flip(HR_Label_patch, axis=2)

        # randomly rotation
        #rotation_degree = random.randint(0, 3)
        #self.data[index, :, :, :] = np.rot90(LR_patch, rotation_degree, (1,2))
        #self.target[index, :, :, :] = np.rot90(HR_patch, rotation_degree, (1,2))
        return LR_patch.copy(), HR_patch.copy() , Label_patch.copy(), \
               HR_Label_patch.copy()

    def __len__(self):
        return self.data.shape[0]

class DatasetFromHdf5_dual(data.Dataset):    #_dual什么意思？ 二重的双的 颜色不变？
    def __init__(self, file_path):
        super(DatasetFromHdf5_dual, self).__init__()
        hf = h5py.File(file_path)
        self.data = hf.get("lr")
        self.target = hf.get("hr")
        self.label = hf.get("label")
        self.hr_label = hf.get("hr_label")
        is_Test = False

        if is_Test == True :
            for index in range(0,min(100,len(self.data))):
                LR = torch.from_numpy(self.data[index, :, :, :]).float()
                HR = torch.from_numpy(self.target[index, :, :, :]).float()
                LABEL = torch.from_numpy(self.label[index,:,:,:]).float()
                LR = tf.ToPILImage()(LR)
                HR = tf.ToPILImage()(HR)
                LABEL = tf.ToPILImage()(LABEL)
                LR.save('result/LR/{0:04d}.jpg'.format(index))
                HR.save('result/HR/{0:04d}.jpg'.format(index))
                LABEL.save('result/LABEL/{0:04d}.jpg'.format(index))
                print(index)

    def __getitem__(self, index):
        # randomly flip
        #print(index)
        #data shppe: C*H*W
        LR_patch = self.data[index, :, :, :]
        HR_patch = self.target[index, :, :, :]
        Label_patch = self.label[index, :, :, :]
        HR_Label_patch = self.hr_label[index, :, :]

        Label_patch = np.expand_dims(Label_patch, axis=0)
        HR_Label_patch = np.expand_dims(HR_Label_patch, axis=0)
        Label_patch = np.asarray(Label_patch, np.float32)
        HR_Label_patch = np.asarray(HR_Label_patch, np.float32)
        LR_patch = np.asarray(LR_patch, np.float32)
        HR_patch = np.asarray(HR_patch, np.float32)

        LR_patch = LR_patch.transpose(0,2,1)
        HR_patch = HR_patch.transpose(0,2,1)
        Label_patch = Label_patch.transpose(0,2,1)
        HR_Label_patch = HR_Label_patch.transpose(0, 2, 1)

        flip_channel = random.randint(0, 1)
        if flip_channel != 0:
            LR_patch = np.flip(LR_patch, axis=2)
            HR_patch = np.flip(HR_patch, axis=2)
            Label_patch = np.flip(Label_patch, axis=2)
            HR_Label_patch = np.flip(HR_Label_patch, axis=2)
        #self.data[index, :, :, :] = LR_patch
        #self.target[index, :, :, :] = HR_patch
        #self.label[index, :, :, :] = Label_patch
        #self.hr_label[index, :, :, :] = HR_Label_patch
        # randomly rotation
        #rotation_degree = random.randint(0, 3)
        #self.data[index, :, :, :] = np.rot90(LR_patch, rotation_degree, (1,2))
        #self.target[index, :, :, :] = np.rot90(HR_patch, rotation_degree, (1,2))
        return LR_patch.copy(), HR_patch.copy() , Label_patch.copy(), \
               HR_Label_patch.copy()

    def __len__(self):
        return self.data.shape[0]



class DatasetFromHdf5_edge(data.Dataset):
    def __init__(self, file_path):
        super(DatasetFromHdf5_edge, self).__init__()
        hf = h5py.File(file_path)
        self.data = hf.get("lr")
        self.target = hf.get("hr")
        self.label = hf.get("label")
        self.hr_label = hf.get("hr_label")
        is_Test = False


    def __getitem__(self, index):
        # randomly flip
        #print(index)
        #data shppe: C*H*W
        LR_patch = self.data[index, :, :, :]
        HR_patch = self.target[index, :, :, :]
        Label_patch = self.label[index, :, :, :]
        HR_Label_patch = self.hr_label[index, :, :]

        Label_patch = np.expand_dims(Label_patch, axis=0)
        HR_Label_patch = np.expand_dims(HR_Label_patch, axis=0)
        Label_patch = np.asarray(Label_patch, np.float32)
        HR_Label_patch = np.asarray(HR_Label_patch, np.float32)
        LR_patch = np.asarray(LR_patch, np.float32)
        HR_patch = np.asarray(HR_patch, np.float32)

        LR_patch = LR_patch.transpose((0,2,1))
        HR_patch = HR_patch.transpose((0,2,1))
        Label_patch = Label_patch.transpose((0,2,1))
        HR_Label_patch = HR_Label_patch.transpose((0, 2, 1))
        ############Generate edges and blurs###############
        LR_patch_tmp = LR_patch.transpose((1, 2, 0))
        HR_patch_tmp = HR_patch.transpose((1, 2, 0))
        LR_patch_blur = ndimage.gaussian_filter(LR_patch_tmp, sigma=(2, 2, 0), order=0)
        HR_patch_blur = ndimage.gaussian_filter(HR_patch_tmp, sigma=(2, 2, 0), order=0)
        LR_patch_edge = LR_patch_tmp - LR_patch_blur
        HR_patch_edge = HR_patch_tmp - HR_patch_blur
        LR_patch_blur = LR_patch_blur.transpose((2, 1, 0))
        HR_patch_blur = HR_patch_blur.transpose((2, 1, 0))
        LR_patch_edge = LR_patch_edge.transpose((2, 1, 0))
        HR_patch_edge = HR_patch_edge.transpose((2, 1, 0))
        '''''''''
        plt.imshow(HR_patch_tmp, interpolation='bilinear')
        plt.axis('off')
        plt.show()
        plt.imshow(HR_patch_blur, interpolation='bilinear')
        plt.axis('off')
        plt.show()
        plt.imshow(HR_patch_edge, interpolation='bilinear')
        plt.axis('off')
        plt.show()
        '''''''''
        flip_channel = random.randint(0, 1)
        if flip_channel != 0:
            LR_patch = np.flip(LR_patch, axis=2)
            HR_patch = np.flip(HR_patch, axis=2)
            LR_patch_blur = np.flip(LR_patch_blur, axis=2)
            HR_patch_blur = np.flip(HR_patch_blur, axis=2)
            LR_patch_edge = np.flip(LR_patch_edge, axis=2)
            HR_patch_edge = np.flip(HR_patch_edge, axis=2)
            Label_patch = np.flip(Label_patch, axis=2)
            HR_Label_patch = np.flip(HR_Label_patch, axis=2)

        # randomly rotation
        #rotation_degree = random.randint(0, 3)
        #self.data[index, :, :, :] = np.rot90(LR_patch, rotation_degree, (1,2))
        #self.target[index, :, :, :] = np.rot90(HR_patch, rotation_degree, (1,2))
        return LR_patch.copy(), \
               HR_patch.copy() , \
               Label_patch.copy(), \
               HR_Label_patch.copy(), \
               LR_patch_blur.copy(), \
               HR_patch_blur.copy(), \
               LR_patch_edge.copy(), \
               HR_patch_edge.copy()

    def __len__(self):
        return self.data.shape[0]