import torch
from torch import nn
import time

class BasicModule(nn.Module):

    '''
    功能： 封装了nn.module,主要提供save和load两个方法
    '''

    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name = str(type(self))

    def load(self, path):
        '''
        功能：可加载指定路径的模型
        :return:
        '''

        self.load_state_dict(torch.load(path))

    def save(self, name=None):
        '''
        功能：保存模型，默认使用“模型名字+时间”作为文件名
        如：AlexNet_1022_23:23:23.pth
        :param name:
        :return:
        '''

        if name is None:
            prefix = 'checkpoints/' + self.model_name + '_'
            name   = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        torch.save(self.state_dict(), name)
        return name