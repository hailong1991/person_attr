import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models

from models.BasicModule import BasicModule

class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

class InceptionAux(nn.Module):

    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.conv0 = BasicConv2d(in_channels, 128, kernel_size=1)
        self.conv1 = BasicConv2d(128, 768, kernel_size=5)
        self.conv1.stddev = 0.01

        self.fc_gender = nn.Linear(768, num_classes['gender'])
        self.fc_hat = nn.Linear(768, num_classes['hat'])
        self.fc_glasses = nn.Linear(768, num_classes['glasses'])
        self.fc_longcoat = nn.Linear(768, num_classes['longcoat'])
        self.fc_boots = nn.Linear(768, num_classes['boots'])
        self.fc_age = nn.Linear(768, num_classes['age'])
        self.fc_position = nn.Linear(768, num_classes['position'])
        self.fc_bag = nn.Linear(768, num_classes['bag'])
        self.fc_sleeve = nn.Linear(768, num_classes['sleeve'])
        self.fc_trousers = nn.Linear(768, num_classes['trousers'])

        # self.fc = nn.Linear(768, num_classes)
        # self.fc.stddev = 0.001

        self.AvgPool2d = nn.AvgPool2d(kernel_size=13, stride=1, padding=0)

    def forward(self, x):
        # 17 x 17 x 768
        x = F.avg_pool2d(x, kernel_size=5, stride=3)
        # 5 x 5 x 768
        x = self.conv0(x)
        # 5 x 5 x 128
        x = self.conv1(x)

        #print(x.size()) #
        x = self.AvgPool2d(x)
        # 1 x 1 x 768
        x = x.view(x.size(0), -1)

        gender_predict = self.fc_gender(x)
        hat_predict = self.fc_hat(x)
        glasses_predict = self.fc_glasses(x)
        longcoat_predict = self.fc_longcoat(x)
        boots_predict = self.fc_boots(x)
        age_predict = self.fc_age(x)
        position_predict = self.fc_position(x)
        bag_predict = self.fc_bag(x)
        sleeve_predict = self.fc_sleeve(x)
        trousers_predict = self.fc_trousers(x)

        return gender_predict, hat_predict, glasses_predict, longcoat_predict, boots_predict, \
               age_predict, position_predict, bag_predict, sleeve_predict, trousers_predict

        # # 768
        # x = self.fc(x)
        # # 1000
        # return x


class PersonInception_v3(BasicModule):

    '''
    功能：以resnet为基础网络的多标签分类网络
    '''

    def __init__(self, class_info, pretrained=False):
        super(PersonInception_v3, self).__init__()
        inception_v3 = models.inception_v3(pretrained=pretrained)
        #print(inception_v3)
        # 最后那层去掉
        # print(inception_v3._modules)
        print(inception_v3._modules.keys())
        # print(inception_v3._modules.values())
        #exit(0)
        print(inception_v3._modules['AuxLogits']._modules['fc'])
        del inception_v3._modules['AuxLogits']._modules['fc']
        inception_v3._modules['AuxLogits']._modules['AuxAvgPool2d'] = nn.AvgPool2d(kernel_size=13, stride=1, padding=0)
        inception_v3._modules['AuxLogits']._modules['fc'] = nn.Linear(in_features=768, out_features=1000, bias=True)
        del inception_v3._modules['fc']  # 网络的models是一个OrderedDict类型，所以将最后那层删除就行了

        #self.backone = nn.Sequential(*list(inception_v3._modules.values())[:-1])

        inception_v3_struct = list(inception_v3._modules.values())
        self.backone = nn.Sequential(*inception_v3_struct[0:13])
        print(self.backone)
        # self.inception_v3_part1 = nn.Sequential(inception_v3_struct[13])
        # #self.inception_v3_part1.add_module('AvgPool2d', nn.AvgPool2d(kernel_size=26, stride=1, padding=0), )
        # #self.inception_v3_part1.add_module('fc', nn.Linear(in_features=768, out_features=1000, bias=True))
        #
        # print(self.inception_v3_part1)

        self.inception_v3_part1 = InceptionAux(768, class_info)
        self.inception_v3_part2 = nn.Sequential(*inception_v3_struct[14:])
        print(self.inception_v3_part2)

        #self.AuxAvgPool2d = nn.AvgPool2d(kernel_size=26, stride=1, padding=0)
        self.AvgPool2d = nn.AvgPool2d(kernel_size=26, stride=1, padding=0)

        # 添加最后的自定义分类层

        #self.aux_fc_gender = nn.Linear(2048, class_info['gender'])

        self.fc_gender = nn.Linear(2048, class_info['gender'])
        self.fc_hat = nn.Linear(2048, class_info['hat'])
        self.fc_glasses = nn.Linear(2048, class_info['glasses'])
        self.fc_longcoat = nn.Linear(2048, class_info['longcoat'])
        self.fc_boots = nn.Linear(2048, class_info['boots'])
        self.fc_age = nn.Linear(2048, class_info['age'])
        self.fc_position = nn.Linear(2048, class_info['position'])
        self.fc_bag = nn.Linear(2048, class_info['bag'])
        self.fc_sleeve = nn.Linear(2048, class_info['sleeve'])
        self.fc_trousers = nn.Linear(2048, class_info['trousers'])

        # finetune
        # if pretrained:
        #     model_path = '/home/oeasy/disk/home/oeasy/lhl/person_attr/person_attr_pytorch/checkpoints/pretrained_model/resnet50.pth'
        #     print("Loading pretrained weights from %s" % (model_path))
        #     state_dict = torch.load(model_path)
        #     # temp = {k: v for k, v in state_dict.items() if k in resnet50.state_dict()}
        #     # self.backone.load_state_dict(temp)
        #     self.backone.load_state_dict({k: v for k, v in state_dict.items() if k in resnet50.state_dict()})

    def forward(self, x):
        x = self.backone(x)

        aux_predict = self.inception_v3_part1(x)

        x = self.inception_v3_part2(x)
        x = self.AvgPool2d(x)
        x = x.view(x.size()[0], -1)
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
               age_predict, position_predict, bag_predict, sleeve_predict, trousers_predict, aux_predict

if __name__ == '__main__':

    labels_dict = {'gender':2, 'hat':2, 'glasses':2, 'longcoat':2, 'boots':2, \
                   'age':3, 'position':3, 'bag':4, 'sleeve':2, 'trousers':3, }
    inception_v3 = PersonInception_v3(labels_dict, pretrained=True)
    print(inception_v3)
    for name, params in inception_v3.named_parameters():
        print(name, '----', params.size())

    input = torch.autograd.Variable(torch.randn(1, 3, 224, 224))
    out = inception_v3(input)
    print(out)