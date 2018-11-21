import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models

from models.BasicModule import BasicModule

class PersonInception_v3(BasicModule):

    '''
    功能：以resnet为基础网络的多标签分类网络
    '''

    def __init__(self, class_info, pretrained=False):
        super(PersonInception_v3, self).__init__()
        inception_v3 = models.inception_v3(pretrained=pretrained)
        # 最后那层去掉
        print(inception_v3._modules.keys())
        del inception_v3._modules['AuxLogits']
        del inception_v3._modules['fc']  # 网络的models是一个OrderedDict类型，所以将最后那层删除就行了

        self.backone = nn.Sequential(*list(inception_v3._modules.values())[:-1])
        self.AvgPool2d = nn.AvgPool2d(kernel_size=26, stride=1, padding=0)
        # 添加最后的自定义分类层
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


    def forward(self, x):
        x = self.backone(x)

        print(type(self.backone[13]), type(self.backone[13].conv1))
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
               age_predict, position_predict, bag_predict, sleeve_predict, trousers_predict

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