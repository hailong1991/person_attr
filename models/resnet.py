import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models

from models.BasicModule import BasicModule

class PersonResnet50(BasicModule):

    '''
    功能：以resnet为基础网络的多标签分类网络
    '''

    def __init__(self, class_info, pretrained=False):
        super(PersonResnet50, self).__init__()
        resnet50 = models.resnet50(pretrained=pretrained)
        # for param in resnet50.parameters():
        #     param.requires_grad = False
        # for name, params in resnet50.named_parameters():
        #     print(name, '----', params.size())
        # 最后那层去掉
        # print(resnet50._modules)
        # print(type(resnet50._modules))
        # print(resnet50._modules.keys())
        # print(resnet50._modules.values())
        #del resnet50._modules['fc']  # 网络的models是一个OrderedDict类型，所以将最后那层删除就行了

        self.backone = nn.Sequential(*list(resnet50._modules.values())[:-1])
        #self.backone = nn.Sequential(resnet50._modules)
        #print(self.backone)

        # 添加最后的自定义分类层
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(2048, 2048),
            torch.nn.BatchNorm1d(2048),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),

        )


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
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
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
    resnet50 = PersonResnet50(labels_dict, pretrained=True)
    print(resnet50)
    for name, params in resnet50.named_parameters():
        print(name, '----', params.size())

    input = torch.autograd.Variable(torch.randn(2, 3, 224, 224))
    out = resnet50(input)
    print(out)