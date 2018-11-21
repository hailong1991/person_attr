import warnings
class DefaultConfig(object):
    '''
    功能：配置文件类
    '''

    env = 'lhl' # visdom 环境
    model = 'vgg16' # 模型
    train_data_root = '/home/oeasy/lhl/caffe/data/person_attri/final/all_train'
    test_data_root = '/home/oeasy/lhl/caffe/data/person_attri/final/test'
    load_model_path = False  #'checkpoints/model.pth' # 加载模型路径

    batchsize  = 30  # 6
    use_gpu = True
    num_workers = 4 # 加载数据的工作线程
    print_freq = 20 # 每N batch打印一次
    debug_file = '/tmp/debug'
    result_file = 'result.csv'
    class_num  = 10 # 10个标签

    max_epoch = 20
    lr = 0.001
    gamma = 0.95 #when val_loss increase , lr = lr*gamma
    weight_decay = 1e-4
    step_size = 10
    decay_type='step'


    def parse(self, kwargs):
        '''
        功能：根据字典kwargs更新config参数
        :param kwargs:
        :return:
        '''

        # 更新配置参数
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning:opt has not attribut %s" %k)
            setattr(self, k, v)

        # 打印配置信息、
        print('user config:')

        for k, v in self.__class__.__dict__.items():
            if not k.startswith('__'):
                print(k, getattr(self, k))