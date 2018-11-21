#coding:utf8
import visdom
import time
import numpy as np

class Visualizer(object):
    '''
    功能：   封装了visdom的基本操作，但是你仍然可以通过`self.vis.function`
           或者`self.function`调用原生的visdom接口
           比如
           self.text('hello visdom')
           self.histogram(t.randn(1000))
           self.line(t.arange(0, 10),t.arange(1, 11))
    '''

    def __init__(self, env='default', **kwargs):
        self.vis = visdom.Visdom(env=env, **kwargs)

        # 画的第几个数，相当于横坐标
        # 比如（’loss',23） 即loss的第23个点
        self.index = {}
        self.log_text = ''

    def reinit(self, env='default', **kwargs):
        '''
        功能： 修改visdom的配置
        :param env:
        :param kwargs:
        :return:
        '''

        self.vis = visdom.Visdom(env='default', **kwargs)
        return self

    def plot_many(self, dict):
        '''
        功能： 一次plot多个
        :param dict:dict (name, value) i.e. ('loss', 0.11)
        :return:
        '''

        for k,v in dict.items():
            self.plot(k, v)

    def img_many(self, dict):
        for k, v in dict.items():
            self.img(k, v)

    def plot(self, name, y, **kwargs):
        '''
        self.plot('loss', 1.00)
        :param name:
        :param y:
        :param kwargs:
        :return:
        '''

        x = self.index.get(name, 0)
        self.vis.line(Y=np.array([y]), X=np.array([x]),
                      win=(name),
                      opts=dict(title=name),
                      update=None if x == 0 else 'append',
                      **kwargs
                      )

        self.index[name] = x + 1

    def img(self, name, img_, **kwargs):
        '''
        self.img('input_img', t.Tensor(64, 64))
        self.img('input_imgs', t.Tensor(3, 64, 64))
        self.img('input_imgs', t.Tensor(100, 1, 64, 64))
        self.img('input_imgs', t.Tensor(100, 3, 64, 64), nrows=10)
        '''
        self.vis.images(img_.cpu().numpy(),
                        win=(name),
                        opts=dict(title=name),
                        **kwargs
                        )

    def log(self, info, win='log_text'):
        '''
        self.log({'loss':1, 'lr':0.0001})
        '''

        self.log_text += ('[{time}] {info} <br>'.format(
            time=time.strftime('%m%d_%H%M%S'), \
            info=info))
        self.vis.text(self.log_text, win)

    def __getattr__(self, name):
        '''
        self.function 等价于self.vis.function
        自定义的plot,image,log,plot_many等除外
        '''
        return getattr(self.vis, name)

    def plot_many_stack(self, d):
        '''
        self.plot('loss',1.00)
        '''
        name=list(d.keys())
        name_total=" ".join(name)
        x = self.index.get(name_total, 0)
        val=list(d.values())
        if len(val)==1:
            y=np.array(val)
        else:
            y=np.array(val).reshape(-1,len(val))
        #print(x)
        self.vis.line(Y=y,X=np.ones(y.shape)*x,
                    win=str(name_total),#unicode
                    opts=dict(legend=name,
                        title=name_total),
                    update=None if x == 0 else 'append'
                    )
        self.index[name_total] = x + 1