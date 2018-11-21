import os
from PIL import Image
from torch.utils import data
import numpy as np
from torchvision import transforms as T

import torch

class PersonDataBinary(data.Dataset):

    def __init__(self, root, transforms=None, test=False):
        '''
        功能：获取所有图片地址，并根据训练、验证、测试划分数据
        :param root: 图片路径
        :param transforms: 是否对图片增强处理
        :param train: 获取训练图片
        :param test: 获取测试、验证图片
        '''
        # root: /home/oeasy/lhl/caffe/data/person_attri/final/
        self.imgs = []  # [xxx.jpg, xxx.jpg, ...]
        self.labels = []  # [[0 1 3 3 3 3 1], ...]
        self.test = test

        normallize = T.Normalize(mean = [0.485, 0.456, 0.406],
                                 std  = [0.229, 0.224, 0.225])

        if transforms == None:
            if self.test:
                self.transforms = T.Compose([
                    T.Resize((224, 224)),  # 缩放384 128
                    #T.CenterCrop(224),  # 裁剪固定大小
                    T.ToTensor(),
                    normallize
                ])
            else:
                self.transforms = T.Compose([
                    T.Resize((224, 224)),  # 缩放
                    T.ToTensor(),
                    normallize
                ])
                # self.transforms = T.Compose([
                #     T.Resize(256), # 缩放
                #     T.RandomResizedCrop(224), # 裁剪固定大小
                #     T.ToTensor(),
                #     normallize
                # ])
        else:
            self.transforms = transforms

        # 读取标注文件获取图片和标签
        if self.test:
            labels_file = os.path.join(root, "binary_test_label.txt")
        else:
            labels_file = os.path.join(root, "binary_train_90000_label.txt")

        with open(labels_file, 'r') as outfile:
            for line in outfile:
                line_content = line.strip('\n\t').split(' ')
                self.imgs.append(line_content[0])
                label = [int(item) for item in line_content[1:]]
                #self.labels.append(np.array(label))
                self.labels.append(torch.FloatTensor(label))

        if self.test:
            self.imgs = [os.path.join(root, 'test/'+img) for img in self.imgs]
        else:
            self.imgs = [os.path.join(root, 'all_train/'+img) for img in self.imgs]


    def __getitem__(self, index):
        '''
        功能： 返回一张图片的id
        :param index:
        :return:
        '''

        img_path = self.imgs[index]
        label = self.labels[index]
        data = Image.open(img_path)
        data = self.transforms(data)
        return data, label

    def __len__(self):
        '''
        功能： 返回数据集中所有图片的个数
        :return:
        '''

        return len(self.imgs)

if __name__ == '__main__':

    import visdom
    import cv2
    vis = visdom.Visdom(env='lhl')
    img_path ='/home/oeasy/lhl/caffe/data/person_attri/final/test/090002.jpg'
    np_img = cv2.imread(img_path)
    np_img = cv2.resize(np_img, (224, 224))
    np_img = np_img.transpose(2,1,0)

    vis.images(np_img, win='source', opts={'title': 'source'})

    transforms = T.Compose([
        T.Resize((200, 200)),  # 缩放
        T.RandomResizedCrop(224),  # 裁剪固定大小
        T.ColorJitter(brightness=0.5),
        T.ToTensor()
    ])
    train_dataset  = PersonDataBinary('/home/oeasy/lhl/caffe/data/person_attri/final/', transforms=transforms, test=True)
    train_loader = data.DataLoader(train_dataset,
                                   batch_size = 3,
                                   shuffle = False,
                                   num_workers = 0)
    for ii, (data, label) in enumerate(train_loader):
        print("------------")
        # print(data.shape)
        # vis.images(data.squeeze(0).numpy(), win='test', opts={'title':'test'})
        # exit(0)

        print('-------------')
        print(label)
        print(label[:, 0])
        print(label[:, 1])
        print(label[:, 2])
        print(label[:, 3])
        print(label[:, 4])
        #label = [[1,2,3],[3,4,5]]
        #print(type(label))
        # a = torch.LongTensor(label)
        # print(a)
        break
