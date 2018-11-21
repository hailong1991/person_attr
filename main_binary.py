import visdom
import torch
from data import dataset_binary
from torch.autograd import Variable
from torch.utils import data
import torch.optim as optim

import utils.visual_loss as visual_loss
import utils.util as utils
import utils.checkpoint as checkpoint
from torchnet import meter

import models
import config
from models import renet_mgn
import models.resnet_binary as resnet_binary
import time
import numpy as np

'''
模型训练、测试接口
'''

class BaseTrain(object):
    def __init__(self, net, train_data_set, test_data_set, opt):
        self.opt = opt
        self.vis = visual_loss.Visualizer(opt.env)
        self.loss_meter = meter.AverageValueMeter()
        self.accuracy_meter = meter.AverageValueMeter()
        self.Predict_meter = meter.AverageValueMeter()
        self.Recall_meter = meter.AverageValueMeter()
        self.F_measure_meter = meter.AverageValueMeter()

        self.use_gpu = opt.use_gpu
        self.device = torch.device("cuda" if self.use_gpu else "cpu")
        self.epoch = opt.max_epoch
        self.batch_size = opt.batchsize
        self.model = net.to(self.device) if self.use_gpu else net
        self.loss_func = torch.nn.MultiLabelSoftMarginLoss()
        #self.optimizer = optim.SGD(self.model.parameters, lr = opt.lr, weight_decay=opt.weight_decay)
        self.optimizer = optim.Adam(self.model.parameters(),
                                                lr = opt.lr,
                                                weight_decay = opt.weight_decay)
        self.train_data_loader = torch.utils.data.DataLoader(train_data_set, self.batch_size,
                                                             shuffle=True,
                                                             num_workers=opt.num_workers)
        self.test_data_loader = torch.utils.data.DataLoader(test_data_set, self.batch_size,
                                                             shuffle=True,
                                                             num_workers=opt.num_workers)
        self.labels = ['Female', 'AgeOver60', 'Age18-60', 'AgeLess18', 'Front', 'Side', 'Back',
                       'Hat', 'Glasses', 'HandBag', 'ShoulderBag', 'Backpack','ShortSleeve',
                       'LongSleeve', 'LongCoat', 'Trousers', 'Shorts', 'Skirt&Dress', 'boots']
    def train(self):

        # 加载模型
        if self.opt.load_model_path:
            self.model.load(self.opt.load_model_path)
        for epoch in range(self.opt.max_epoch):
            self.loss_meter.reset()
            self.train_one_epoch(epoch)

        self.model.save()

    def train_one_epoch(self, epoch):
        self.model.train()

        print('Begin training...')
        for batch_index, (img, label) in enumerate(self.train_data_loader, 0):

            # 训练参数模型
            input = Variable(img)
            target = Variable(label)
            if self.opt.use_gpu:
                input, target = input.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(input)
            loss = self.loss_func(outputs, target)
            loss.backward()
            self.optimizer.step()

            self.loss_meter.add(loss.data.item())
            if batch_index % self.opt.print_freq == self.opt.print_freq-1:
                self.vis.plot_many({'train_loss': self.loss_meter.value()[0]})



            if batch_index % 200 == 0:
                self.test(batch_index)


    def test(self, epoch):
        self.model.eval()
        print('Begin test...')
        self.accuracy_meter.reset()
        self.Predict_meter.reset()
        self.Recall_meter.reset()
        self.F_measure_meter.reset()

        attr_count = 0
        with torch.no_grad():
            for batch_index, (img, label) in enumerate(self.test_data_loader, 0):
                input = Variable(img)
                labels = Variable(label)
                if self.opt.use_gpu:
                    input = input.to(self.device)
                outputs = self.model(input)


                #if batch_index % 100 == 0:
                y_true = labels.cpu().numpy()
                y_pred = outputs.cpu().detach().numpy()
                y_pred = utils.sigmoid(y_pred)
                accuracy = utils.accuracy(y_true, y_pred)
                gender_count =  utils.attr_accuracy(y_true, y_pred, 0)
                attr_count += gender_count
                #print('xxxx attr_count', attr_count)
                predict = utils.precision(y_true, y_pred)
                recall = utils.recall(y_true, y_pred)
                fmeasure = utils.fmeasure(predict, recall)

                self.accuracy_meter.add(accuracy)
                self.Predict_meter.add(predict)
                self.Recall_meter.add(recall)
                self.F_measure_meter.add(fmeasure)

            print('index: [%d]' % (len(self.test_data_loader)),
                  'Train epoch: [%d]' % epoch,
                  'accuracy:%.6f' % accuracy,
                  'Predict:%.6f' % predict,
                  'Recall:%.6f' % recall,
                  'F-measure:%.6f' % fmeasure)

        self.vis.plot_many({'accuracy': self.accuracy_meter.value()[0]})
        self.vis.plot_many({'predict': self.Predict_meter.value()[0]})
        self.vis.plot_many({'recall': self.Recall_meter.value()[0]})
        self.vis.plot_many({'fmeasure': self.F_measure_meter.value()[0]})

        gender_accuracy =  attr_count / 10000.0
        print('attr_count', attr_count)
        self.vis.plot('gender_accuracy', gender_accuracy)

def train(**kwargs):

    opt = config.DefaultConfig()
    # 根据命令行输入参数
    print('---------------------------------------')
    opt.parse(kwargs)
    print('---------------------------------------')
    root = '/home/oeasy/lhl/caffe/data/person_attri/final/'
    train_data_set = dataset_binary.PersonDataBinary(root)
    test_data_set  = dataset_binary.PersonDataBinary(root, test=True)

    model = resnet_binary.PersonResnet50Binary(19, pretrained=True)
    train = BaseTrain(model, train_data_set, test_data_set, opt)
    train.train()

if __name__=='__main__':
    train()