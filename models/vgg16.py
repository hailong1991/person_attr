import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models

from models.BasicModule import BasicModule

class PersonVgg(BasicModule):

    '''
    功能：以vgg16为基础网络的多标签分类网络
    '''

    def __init__(self, class_info, pretrained=False):
        super(PersonVgg, self).__init__()
        vgg16 = models.vgg16_bn(pretrained=False)

        # fixed
        # for parma in vgg16.parameters():
        #     parma.requires_grad = False

        # 最后那层去掉
        vgg16.classifier = nn.Sequential(*list(vgg16.classifier._modules.values())[:-1])

        # finetune
        if pretrained:
            model_path = '/home/oeasy/disk/home/oeasy/lhl/person_attr/person_attr_pytorch/checkpoints/pretrained_model/vgg16_bn.pth'
            print("Loading pretrained weights from %s" % (model_path))
            state_dict = torch.load(model_path)
            vgg16.load_state_dict({k: v for k, v in state_dict.items() if k in vgg16.state_dict()})

        self.backone     = vgg16

        # 添加最后的自定义分类层
        self.fc_gender   = nn.Linear(4096, class_info['gender'])
        self.fc_hat      = nn.Linear(4096, class_info['hat'])
        self.fc_glasses  = nn.Linear(4096, class_info['glasses'])
        self.fc_longcoat = nn.Linear(4096, class_info['longcoat'])
        self.fc_boots    = nn.Linear(4096, class_info['boots'])
        self.fc_age      = nn.Linear(4096, class_info['age'])
        self.fc_position = nn.Linear(4096, class_info['position'])
        self.fc_bag      = nn.Linear(4096, class_info['bag'])
        self.fc_sleeve   = nn.Linear(4096, class_info['sleeve'])
        self.fc_trousers = nn.Linear(4096, class_info['trousers'])

        # 初始化参数, 其实nn.Linear会默认初始化，这里还是显示初始化下
        self._init_fc(self.fc_gender)
        self._init_fc(self.fc_hat)
        self._init_fc(self.fc_glasses)
        self._init_fc(self.fc_longcoat)
        self._init_fc(self.fc_boots)
        self._init_fc(self.fc_age)
        self._init_fc(self.fc_position)
        self._init_fc(self.fc_bag)
        self._init_fc(self.fc_sleeve)
        self._init_fc(self.fc_trousers)

    @staticmethod
    def _init_fc(fc):
        '''
        功能：初始化全连接层
        :param fc:
        :return:
        '''
        #nn.init.kaiming_normal_(fc.weight, mode='fan_out')
        nn.init.normal_(fc.weight, std=0.001)
        nn.init.constant_(fc.bias, 0.)

    def forward(self, x):
        x = self.backone(x)

        gender_predict   = self.fc_gender(x)
        hat_predict      = self.fc_hat(x)
        glasses_predict  = self.fc_glasses(x)
        longcoat_predict = self.fc_longcoat(x)
        boots_predict    = self.fc_boots(x)
        age_predict      = self.fc_age(x)
        position_predict = self.fc_position(x)
        bag_predict      = self.fc_bag(x)
        sleeve_predict   = self.fc_sleeve(x)
        trousers_predict = self.fc_trousers(x)

        return gender_predict, hat_predict, glasses_predict, longcoat_predict, boots_predict, \
               age_predict, position_predict, bag_predict, sleeve_predict, trousers_predict

if __name__ == '__main__':

    labels_dict = {'gender':2, 'hat':2, 'glasses':2, 'longcoat':2, 'boots':2, \
                   'age':3, 'position':3, 'bag':4, 'sleeve':2, 'trousers':3, }
    vgg16 = PersonVgg(labels_dict, pretrained=True)
    print(vgg16)
    for name, params in vgg16.named_parameters():
        print(name, '----', params.size())
    input = torch.autograd.Variable(torch.randn(1, 3, 224, 224))
    out = vgg16(input)
    print(out)