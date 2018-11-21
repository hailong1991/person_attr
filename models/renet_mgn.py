import copy

import torch
from torch import nn
import torch.nn.functional as F

from torchvision.models.resnet import resnet50, Bottleneck


def make_model(args):
    return MGN(args)

class ResnetAux(nn.Module):

    def __init__(self, in_channels, num_classes):
        super(ResnetAux, self).__init__()

        self.avgpool = nn.AvgPool2d(kernel_size=(24, 8))

        self.fc_gender = nn.Linear(in_channels, num_classes['gender'])
        self.fc_hat = nn.Linear(in_channels, num_classes['hat'])
        self.fc_glasses = nn.Linear(in_channels, num_classes['glasses'])
        self.fc_longcoat = nn.Linear(in_channels, num_classes['longcoat'])
        self.fc_boots = nn.Linear(in_channels, num_classes['boots'])
        self.fc_age = nn.Linear(in_channels, num_classes['age'])
        self.fc_position = nn.Linear(in_channels, num_classes['position'])
        self.fc_bag = nn.Linear(in_channels, num_classes['bag'])
        self.fc_sleeve = nn.Linear(in_channels, num_classes['sleeve'])
        self.fc_trousers = nn.Linear(in_channels, num_classes['trousers'])

    def forward(self, x):
        # 1024 x 24 x 8
        x = self.avgpool(x)
        # 1 x 1 x 1024
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


class MGN(nn.Module):
    def __init__(self, class_info, pretrained=False):
        super(MGN, self).__init__()

        resnet = resnet50(pretrained=pretrained)

        self.backone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
        )

        self.res_conv4 = nn.Sequential(*resnet.layer3[1:])

        res_g_conv5 = resnet.layer4

        res_p_conv5 = nn.Sequential(
            Bottleneck(1024, 512, downsample=nn.Sequential(nn.Conv2d(1024, 2048, 1, bias=False), nn.BatchNorm2d(2048))),
            Bottleneck(2048, 512),
            Bottleneck(2048, 512))
        res_p_conv5.load_state_dict(resnet.layer4.state_dict())

        self.p1 = nn.Sequential(copy.deepcopy(self.res_conv4), copy.deepcopy(res_g_conv5))
        self.p2 = nn.Sequential(copy.deepcopy(self.res_conv4), copy.deepcopy(res_p_conv5))
        self.p3 = nn.Sequential(copy.deepcopy(res_p_conv5))

        # if args.pool == 'max':
        #     pool2d = nn.MaxPool2d
        # elif args.pool == 'avg':
        #     pool2d = nn.AvgPool2d
        # else:
        #     raise Exception()
        pool2d = nn.AvgPool2d

        self.maxpool_zg_p1 = pool2d(kernel_size=(12, 4))
        self.maxpool_zg_p2 = pool2d(kernel_size=(24, 8))
        self.maxpool_zg_p3 = pool2d(kernel_size=(24, 8))
        self.maxpool_zp2 = pool2d(kernel_size=(12, 8))
        self.maxpool_zp3 = pool2d(kernel_size=(8, 8))
        self.ResnetAux = ResnetAux(1024, class_info)

        reduction = nn.Sequential(nn.Conv2d(2048, 256, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU())

        self._init_reduction(reduction)
        self.reduction_0 = copy.deepcopy(reduction)
        self.reduction_1 = copy.deepcopy(reduction)
        self.reduction_2 = copy.deepcopy(reduction)
        self.reduction_3 = copy.deepcopy(reduction)
        self.reduction_4 = copy.deepcopy(reduction)
        self.reduction_5 = copy.deepcopy(reduction)
        self.reduction_6 = copy.deepcopy(reduction)
        self.reduction_7 = copy.deepcopy(reduction)

        # self.fc_id_2048_0 = nn.Linear(2048, num_classes)
        #self.fc_id_2048_0 = nn.Linear(256, num_classes)
        self.fc_gender = nn.Linear(6144, class_info['gender'])
        self.fc_hat = nn.Linear(4096, class_info['hat'])
        self.fc_glasses = nn.Linear(4096, class_info['glasses'])
        self.fc_longcoat = nn.Linear(6144, class_info['longcoat'])
        self.fc_boots = nn.Linear(4096, class_info['boots'])
        self.fc_age = nn.Linear(4096, class_info['age'])
        self.fc_position = nn.Linear(4096, class_info['position'])
        self.fc_bag = nn.Linear(6144, class_info['bag'])
        self.fc_sleeve = nn.Linear(6144, class_info['sleeve'])
        self.fc_trousers = nn.Linear(6144, class_info['trousers'])

        # self.fc_id_2048_1 = nn.Linear(args.feats, num_classes)
        # self.fc_id_2048_2 = nn.Linear(args.feats, num_classes)
        #
        # self.fc_id_256_1_0 = nn.Linear(args.feats, num_classes)
        # self.fc_id_256_1_1 = nn.Linear(args.feats, num_classes)
        # self.fc_id_256_2_0 = nn.Linear(args.feats, num_classes)
        # self.fc_id_256_2_1 = nn.Linear(args.feats, num_classes)
        # self.fc_id_256_2_2 = nn.Linear(args.feats, num_classes)
        #
        # self._init_fc(self.fc_id_2048_0)
        # self._init_fc(self.fc_id_2048_1)
        # self._init_fc(self.fc_id_2048_2)
        #
        # self._init_fc(self.fc_id_256_1_0)
        # self._init_fc(self.fc_id_256_1_1)
        # self._init_fc(self.fc_id_256_2_0)
        # self._init_fc(self.fc_id_256_2_1)
        # self._init_fc(self.fc_id_256_2_2)

    @staticmethod
    def _init_reduction(reduction):
        # conv
        nn.init.kaiming_normal_(reduction[0].weight, mode='fan_in')
        # nn.init.constant_(reduction[0].bias, 0.)

        # bn
        nn.init.normal_(reduction[1].weight, mean=1., std=0.02)
        nn.init.constant_(reduction[1].bias, 0.)

    @staticmethod
    def _init_fc(fc):
        nn.init.kaiming_normal_(fc.weight, mode='fan_out')
        # nn.init.normal_(fc.weight, std=0.001)
        nn.init.constant_(fc.bias, 0.)

    def forward(self, x):

        x = self.backone(x)

        aux_predict = self.ResnetAux(x)



        p3 = self.p3(x)


        zg_p3 = self.maxpool_zg_p3(p3).squeeze(dim=3).squeeze(dim=2) # x.view(x.size()[0], -1)



        zp3 = self.maxpool_zp3(p3)
        z0_p3 = zp3[:, :, 0:1, :].squeeze(dim=3).squeeze(dim=2)
        z1_p3 = zp3[:, :, 1:2, :].squeeze(dim=3).squeeze(dim=2)
        z2_p3 = zp3[:, :, 2:3, :].squeeze(dim=3).squeeze(dim=2)





        gender_predict = torch.cat([zg_p3, z0_p3,  z2_p3], dim=1)
        hat_predict = torch.cat([zg_p3, z0_p3], dim=1)
        glasses_predict = torch.cat([zg_p3,  z0_p3], dim=1)
        longcoat_predict = torch.cat([zg_p3,  z1_p3,  z2_p3], dim=1)
        boots_predict = torch.cat([zg_p3,  z2_p3], dim=1)
        age_predict = torch.cat([zg_p3,  z0_p3], dim=1)
        position_predict = torch.cat([zg_p3, z1_p3], dim=1)
        bag_predict = torch.cat([zg_p3, z0_p3,  z1_p3], dim=1)
        sleeve_predict = torch.cat([zg_p3,  z0_p3,  z1_p3], dim=1)
        trousers_predict = torch.cat([zg_p3, z1_p3, z2_p3], dim=1)
        #
        gender_predict = self.fc_gender(gender_predict)
        hat_predict = self.fc_hat(hat_predict)
        glasses_predict = self.fc_glasses(glasses_predict)
        longcoat_predict = self.fc_longcoat(longcoat_predict)
        boots_predict = self.fc_boots(boots_predict)
        age_predict = self.fc_age(age_predict)
        position_predict = self.fc_position(position_predict)
        bag_predict = self.fc_bag(bag_predict)
        sleeve_predict = self.fc_sleeve(sleeve_predict)
        trousers_predict = self.fc_trousers(trousers_predict)

        return gender_predict, hat_predict, glasses_predict, longcoat_predict, boots_predict, \
               age_predict, position_predict, bag_predict, sleeve_predict, trousers_predict, aux_predict
    #
    # def forward(self, x):
    #
    #     x = self.backone(x)
    #
    #     p1 = self.p1(x)
    #     p2 = self.p2(x)
    #     p3 = self.p3(x)
    #
    #     zg_p1 = self.maxpool_zg_p1(p1)
    #     zg_p2 = self.maxpool_zg_p2(p2)
    #     zg_p3 = self.maxpool_zg_p3(p3)
    #
    #     zp2 = self.maxpool_zp2(p2)
    #     z0_p2 = zp2[:, :, 0:1, :]
    #     z1_p2 = zp2[:, :, 1:2, :]
    #
    #     zp3 = self.maxpool_zp3(p3)
    #     z0_p3 = zp3[:, :, 0:1, :]
    #     z1_p3 = zp3[:, :, 1:2, :]
    #     z2_p3 = zp3[:, :, 2:3, :]
    #
    #     fg_p1 = self.reduction_0(zg_p1).squeeze(dim=3).squeeze(dim=2)
    #     fg_p2 = self.reduction_1(zg_p2).squeeze(dim=3).squeeze(dim=2)
    #     fg_p3 = self.reduction_2(zg_p3).squeeze(dim=3).squeeze(dim=2)
    #     f0_p2 = self.reduction_3(z0_p2).squeeze(dim=3).squeeze(dim=2)
    #     f1_p2 = self.reduction_4(z1_p2).squeeze(dim=3).squeeze(dim=2)
    #     f0_p3 = self.reduction_5(z0_p3).squeeze(dim=3).squeeze(dim=2)
    #     f1_p3 = self.reduction_6(z1_p3).squeeze(dim=3).squeeze(dim=2)
    #     f2_p3 = self.reduction_7(z2_p3).squeeze(dim=3).squeeze(dim=2)
    #
    #     # '''
    #     # l_p1 = self.fc_id_2048_0(zg_p1.squeeze(dim=3).squeeze(dim=2))
    #     # l_p2 = self.fc_id_2048_1(zg_p2.squeeze(dim=3).squeeze(dim=2))
    #     # l_p3 = self.fc_id_2048_2(zg_p3.squeeze(dim=3).squeeze(dim=2))
    #     # '''
    #     # l_p1 = self.fc_id_2048_0(fg_p1)
    #     # l_p2 = self.fc_id_2048_1(fg_p2)
    #     # l_p3 = self.fc_id_2048_2(fg_p3)
    #     #
    #     # l0_p2 = self.fc_id_256_1_0(f0_p2)
    #     # l1_p2 = self.fc_id_256_1_1(f1_p2)
    #     # l0_p3 = self.fc_id_256_2_0(f0_p3)
    #     # l1_p3 = self.fc_id_256_2_1(f1_p3)
    #     # l2_p3 = self.fc_id_256_2_2(f2_p3)
    #
    #     #predict = torch.cat([fg_p1, fg_p2, fg_p3, f0_p2, f1_p2, f0_p3, f1_p3, f2_p3], dim=1)
    #
    #     gender_predict = torch.cat([fg_p1, f0_p2,  f0_p3, f2_p3], dim=1)
    #     hat_predict = torch.cat([fg_p1, f0_p2, f0_p3], dim=1)
    #     glasses_predict = torch.cat([fg_p1,  f0_p2, f0_p3], dim=1)
    #     longcoat_predict = torch.cat([fg_p1,  f1_p2,  f1_p3, f2_p3], dim=1)
    #     boots_predict = torch.cat([fg_p1,  f1_p2, f2_p3], dim=1)
    #     age_predict = torch.cat([fg_p1,  f0_p2,  f0_p3, f1_p3], dim=1)
    #     position_predict = torch.cat([fg_p1, fg_p2, f1_p3, f2_p3], dim=1)
    #     bag_predict = torch.cat([fg_p1, f0_p2, f0_p3, f1_p3], dim=1)
    #     sleeve_predict = torch.cat([fg_p1,  f0_p2,  f0_p3, f1_p3], dim=1)
    #     trousers_predict = torch.cat([fg_p1, f1_p2, f1_p3, f2_p3], dim=1)
    #
    #     gender_predict = self.fc_gender(gender_predict)
    #     hat_predict = self.fc_hat(hat_predict)
    #     glasses_predict = self.fc_glasses(glasses_predict)
    #     longcoat_predict = self.fc_longcoat(longcoat_predict)
    #     boots_predict = self.fc_boots(boots_predict)
    #     age_predict = self.fc_age(age_predict)
    #     position_predict = self.fc_position(position_predict)
    #     bag_predict = self.fc_bag(bag_predict)
    #     sleeve_predict = self.fc_sleeve(sleeve_predict)
    #     trousers_predict = self.fc_trousers(trousers_predict)
    #
    #     return gender_predict, hat_predict, glasses_predict, longcoat_predict, boots_predict, \
    #            age_predict, position_predict, bag_predict, sleeve_predict, trousers_predict
if __name__ == '__main__':

    labels_dict = {'gender':2, 'hat':2, 'glasses':2, 'longcoat':2, 'boots':2, \
                   'age':3, 'position':3, 'bag':4, 'sleeve':2, 'trousers':3, }
    inception_v3 = MGN(labels_dict, pretrained=True)
    print(inception_v3)
    for name, params in inception_v3.named_parameters():
        print(name, '----', params.size())

    input = torch.autograd.Variable(torch.randn(2, 3, 384, 128))
    out = inception_v3(input)
    print(out)