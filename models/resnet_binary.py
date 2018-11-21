import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models

from models.BasicModule import BasicModule

class PersonResnet50Binary(BasicModule):

    '''
    功能：以resnet为基础网络的多标签分类网络
    '''

    def __init__(self, class_info, pretrained=False):
        super(PersonResnet50Binary, self).__init__()
        self.resnet50 = models.resnet50(pretrained=pretrained)

        #self.backone = nn.Sequential(*list(resnet50._modules.values())[:-1])

        # 添加最后的自定义分类层

        self.resnet50.fc = nn.Linear(2048, class_info)



    def forward(self, x):
        predict = self.resnet50(x)
        # x = x.view(x.size()[0], -1)
        # predict   = self.fc_gender(x)
        # hat_predict      = self.fc_hat(x)
        # glasses_predict  = self.fc_glasses(x)
        # longcoat_predict = self.fc_longcoat(x)
        # boots_predict    = self.fc_boots(x)
        # age_predict      = self.fc_age(x)
        # position_predict = self.fc_position(x)
        # bag_predict      = self.fc_bag(x)
        # sleeve_predict   = self.fc_sleeve(x)
        # trousers_predict = self.fc_trousers(x)

        return predict

if __name__ == '__main__':

    labels_dict = {'gender':2, 'hat':2, 'glasses':2, 'longcoat':2, 'boots':2, \
                   'age':3, 'position':3, 'bag':4, 'sleeve':2, 'trousers':3, }
    resnet50 = PersonResnet50Binary(19, pretrained=True)
    print(resnet50)
    for name, params in resnet50.named_parameters():
        print(name, '----', params.size())

    input = torch.autograd.Variable(torch.randn(1, 3, 224, 224))
    out = resnet50(input)
    print(out)