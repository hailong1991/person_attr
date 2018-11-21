from __future__ import absolute_import


import visdom
import torch
from data import dataset
from torch.autograd import Variable
from torch.utils import data

import utils.visual_loss as visual_loss
from torchnet import meter
import torch.optim.lr_scheduler as lrs

import models
import config
from models import renet_mgn

'''
模型训练、测试接口
'''
opt = config.DefaultConfig()


def make_optimizer(opt, type, net, model):
    #trainable = filter(lambda x: x.requires_grad, model.parameters())

    if type == 'SGD':
        optimizer_function = torch.optim.SGD
    elif type == 'ADAM':
        optimizer_function = torch.optim.Adam
    else:
        raise Exception()

    if net == 'vgg':
        trainable = [
            {'params': model.backone.features.parameters()},
            {'params': model.backone.classifier.parameters()},
            {'params': model.fc_gender.parameters(), 'lr': 30 * opt.lr},
            {'params': model.fc_hat.parameters(), 'lr': 20 * opt.lr},
            {'params': model.fc_glasses.parameters(), 'lr': 20 * opt.lr},
            {'params': model.fc_longcoat.parameters(), 'lr': 20 * opt.lr},
            {'params': model.fc_boots.parameters(), 'lr': 20 * opt.lr},
            {'params': model.fc_age.parameters(), 'lr': 30 * opt.lr},
            {'params': model.fc_position.parameters(), 'lr': 40 * opt.lr},
            {'params': model.fc_bag.parameters(), 'lr': 40 * opt.lr},
            {'params': model.fc_sleeve.parameters(), 'lr': 30 * opt.lr},
            {'params': model.fc_trousers.parameters(), 'lr': 30 * opt.lr}
        ]
    elif net == 'inception_v3':
        trainable = [
            {'params': model.backone.parameters()},
            {'params': model.inception_v3_part2.parameters()},
            {'params': model.inception_v3_part1.parameters(), 'lr': 10 * opt.lr},
            {'params': model.fc_gender.parameters(), 'lr': 30 * opt.lr},
            {'params': model.fc_hat.parameters(), 'lr': 20 * opt.lr},
            {'params': model.fc_glasses.parameters(), 'lr': 20 * opt.lr},
            {'params': model.fc_longcoat.parameters(), 'lr': 20 * opt.lr},
            {'params': model.fc_boots.parameters(), 'lr': 20 * opt.lr},
            {'params': model.fc_age.parameters(), 'lr': 30 * opt.lr},
            {'params': model.fc_position.parameters(), 'lr': 40 * opt.lr},
            {'params': model.fc_bag.parameters(), 'lr': 40 * opt.lr},
            {'params': model.fc_sleeve.parameters(), 'lr': 30 * opt.lr},
            {'params': model.fc_trousers.parameters(), 'lr': 30 * opt.lr}
        ]
    elif net == 'resnet50':
        trainable = [
            {'params': model.backone.parameters()},
            {'params': model.fc.parameters(), 'lr': 10 * opt.lr},
            {'params': model.fc_gender.parameters(), 'lr': 30 * opt.lr},
            {'params': model.fc_hat.parameters(), 'lr': 20 * opt.lr},
            {'params': model.fc_glasses.parameters(), 'lr': 20 * opt.lr},
            {'params': model.fc_longcoat.parameters(), 'lr': 20 * opt.lr},
            {'params': model.fc_boots.parameters(), 'lr': 20 * opt.lr},
            {'params': model.fc_age.parameters(), 'lr': 30 * opt.lr},
            {'params': model.fc_position.parameters(), 'lr': 40 * opt.lr},
            {'params': model.fc_bag.parameters(), 'lr': 40 * opt.lr},
            {'params': model.fc_sleeve.parameters(), 'lr': 30 * opt.lr},
            {'params': model.fc_trousers.parameters(), 'lr': 30 * opt.lr}
        ]
    elif net == 'resnet_mgn':
        trainable = [
            {'params': model.backone.parameters()},
            #{'params': model.p1.parameters()},
            #{'params': model.p2.parameters()},
            {'params': model.p3.parameters()},
            {'params': model.ResnetAux.parameters(), 'lr': 10 * opt.lr},
            {'params': model.fc_gender.parameters(), 'lr': 30 * opt.lr},
            {'params': model.fc_hat.parameters(), 'lr': 20 * opt.lr},
            {'params': model.fc_glasses.parameters(), 'lr': 20 * opt.lr},
            {'params': model.fc_longcoat.parameters(), 'lr': 20 * opt.lr},
            {'params': model.fc_boots.parameters(), 'lr': 20 * opt.lr},
            {'params': model.fc_age.parameters(), 'lr': 30 * opt.lr},
            {'params': model.fc_position.parameters(), 'lr': 40 * opt.lr},
            {'params': model.fc_bag.parameters(), 'lr': 40 * opt.lr},
            {'params': model.fc_sleeve.parameters(), 'lr': 30 * opt.lr},
            {'params': model.fc_trousers.parameters(), 'lr': 30 * opt.lr}
        ]
    else:
        trainable = model.parameters()

    kwargs = {
        'lr': opt.lr,
        'weight_decay': opt.weight_decay
        }

    return optimizer_function(trainable, **kwargs)


def make_scheduler(opt, optimizer):

    if opt.decay_type == 'step':
        scheduler = lrs.StepLR(
            optimizer,
            step_size=opt.step_size,
            gamma=opt.gamma
        )
    elif opt.decay_type.find('step') >= 0:
        milestones = opt.decay_type.split('_')
        milestones.pop(0)
        milestones = list(map(lambda x: int(x), milestones))
        scheduler = lrs.MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=opt.gamma
        )

    return scheduler

def train(**kwargs):
    '''
    功能：训练模型入口
    :param kwargs:
    :return:
    '''
    # 根据命令行输入参数
    print('---------------------------------------')
    opt.parse(kwargs)
    print('---------------------------------------')
    vis = visual_loss.Visualizer(opt.env)

    # step1: 定义加载模型
    label_name = ['gender', 'hat', 'glasses', 'longcoat', 'boots', 'age', 'position', 'bag', 'sleeve', 'trousers']
    labels_dict = {'gender': 2, 'hat': 2, 'glasses': 2, 'longcoat': 2, 'boots': 2, \
                   'age': 3, 'position': 3, 'bag': 4, 'sleeve': 2, 'trousers': 3}
    #model = models.PersonVgg(labels_dict, pretrained=True)
    model = models.PersonResnet50(labels_dict, pretrained=True)
    #model = models.PersonInception_v3(labels_dict, pretrained=True)
    #model = renet_mgn.MGN(labels_dict, pretrained=True)
    print(model)
    #for name, param in model.named_parameters():
    #    print(name, '---', param.size())
    print('---------------------------------------')
    if opt.load_model_path:
        model.load(opt.load_model_path)
    if opt.use_gpu:
        model.cuda()

    # step2: 加载数据
    root =  '/home/oeasy/lhl/caffe/data/person_attri/final/'
    train_data = dataset.PersonData(root)
    test_data  = dataset.PersonData(root, test=True)

    train_data_loader = data.DataLoader(train_data, opt.batchsize,
                                        shuffle=True,
                                        num_workers=opt.num_workers)

    test_data_loader = data.DataLoader(test_data, opt.batchsize,
                                        shuffle=False,
                                        num_workers=opt.num_workers)

    # step3: 定义目标函数和优化器
    weight =torch.FloatTensor([1, 4])
    loss_fun_glass = torch.nn.CrossEntropyLoss(weight=weight.cuda())
    weight = torch.FloatTensor([1, 20])
    loss_fun_hat = torch.nn.CrossEntropyLoss(weight=weight.cuda())
    weight = torch.FloatTensor([1, 30])
    loss_fun_longcoat = torch.nn.CrossEntropyLoss(weight=weight.cuda())
    weight = torch.FloatTensor([1, 50])
    loss_fun_boot = torch.nn.CrossEntropyLoss(weight=weight.cuda())
    weight = torch.FloatTensor([40, 1, 20])
    loss_fun_age = torch.nn.CrossEntropyLoss(weight=weight.cuda())

    loss_fun = torch.nn.CrossEntropyLoss()

    lr = opt.lr

    optimizer = make_optimizer(opt, 'SGD', 'resnet50', model)
    scheduler = make_scheduler(opt, optimizer)

    scheduler.step()
    # epoch = scheduler.last_epoch + 1
    # lr = scheduler.get_lr()[0]

    # step4: 统计指标：平滑处理之后的损失函数，还有混淆矩阵
    loss_meter = meter.AverageValueMeter()
    loss_meter_all = []
    for i in range(opt.class_num):
        loss_meter_all.append(meter.AverageValueMeter())
    previous_loss = 1e100

    # step5:训练
    print('start train.....')
    for epoch in range(opt.max_epoch):
        loss_meter.reset()
        for i in range(opt.class_num):
            loss_meter_all[i].reset()

        for ii, (img, label) in enumerate(train_data_loader):

            # 训练参数模型
            input = Variable(img)
            target = Variable(label)
            if opt.use_gpu:
                input  = input.cuda()
                target = target.cuda()

            optimizer.zero_grad()
            predict = model(input)
            #for i in range(10): # 10表示有10个标签
            # loss = 0.0
            # per_loss = []
            # low_probably_label = ['gender', 'position', 'bag', 'sleeve', 'trousers']
            # for i in range(opt.class_num):
            #     single_loss = loss_fun(predict[i], target[:, i])
            #     per_loss.append(single_loss)
            #     if label_name[i] in low_probably_label:
            #         loss += 10*single_loss
            #     else:
            #         loss += 0.1 * single_loss
            #['gender', 'hat', 'glasses', 'longcoat', 'boots', 'age', 'position', 'bag', 'sleeve', 'trousers']
            per_loss = []
            loss0 = loss_fun(predict[0], target[:, 0])
            loss1 = loss_fun_hat(predict[1], target[:, 1])
            loss2 = loss_fun_glass(predict[2], target[:, 2])
            loss3 = loss_fun_longcoat(predict[3], target[:, 3])
            loss4 = loss_fun_boot(predict[4], target[:, 4])
            loss5 = loss_fun_age(predict[5], target[:, 5])
            loss6 = loss_fun(predict[6], target[:, 6])
            loss7 = loss_fun(predict[7], target[:, 7])
            loss8 = loss_fun(predict[8], target[:, 8])
            loss9 = loss_fun(predict[9], target[:, 9])

            loss = 0.3 * loss0 + 0.1 * loss1 + 0.1 * loss2 + 0.1 * loss3 + 0.1 * loss4 + \
                   0.3 * loss5 + 0.4 * loss6 + 0.3 * loss7 + 0.2 * loss8 + 0.2 * loss9

            # aux_loss0 = loss_fun(predict[10][0], target[:, 0])
            # aux_loss1 = loss_fun(predict[10][1], target[:, 1])
            # aux_loss2 = loss_fun(predict[10][2], target[:, 2])
            # aux_loss3 = loss_fun(predict[10][3], target[:, 3])
            # aux_loss4 = loss_fun(predict[10][4], target[:, 4])
            # aux_loss5 = loss_fun(predict[10][5], target[:, 5])
            # aux_loss6 = loss_fun(predict[10][6], target[:, 6])
            # aux_loss7 = loss_fun(predict[10][7], target[:, 7])
            # aux_loss8 = loss_fun(predict[10][8], target[:, 8])
            # aux_loss9 = loss_fun(predict[10][9], target[:, 9])
            #
            # loss = 0.3*loss0 + 0.1*loss1 + 0.1*loss2 + 0.1*loss3 + 0.1*loss4 + \
            #        0.3*loss5 + 0.4*loss6 + 0.3*loss7 + 0.2*loss8 + 0.2*loss9 + \
            #        0.3*aux_loss0 + 0.1*aux_loss1 + 0.1*aux_loss2 + 0.1*aux_loss3 + 0.1*aux_loss4 + \
            #        0.3*aux_loss5 + 0.4*aux_loss6 + 0.3*aux_loss7 + 0.2*aux_loss8 + 0.2*aux_loss9

            per_loss.append(loss0)
            per_loss.append(loss1)
            per_loss.append(loss2)
            per_loss.append(loss3)
            per_loss.append(loss4)
            per_loss.append(loss5)
            per_loss.append(loss6)
            per_loss.append(loss7)
            per_loss.append(loss8)
            per_loss.append(loss9)

            loss.backward()

            optimizer.step()

            # 更新统计指标及可视化
            loss_meter.add(loss.data.item())
            for i in range(10):
                loss_meter_all[i].add(per_loss[i].data.item())

            if ii%opt.print_freq == opt.print_freq-1:
                vis.plot_many({'train_loss': loss_meter.value()[0]})
                for i in range(opt.class_num):
                    vis.plot_many({'train_loss_'+label_name[i]: loss_meter_all[i].value()[0]})


            if ii % 200 == 200 - 1: #2500
                val_cm, val_accuracy = val(model, test_data_loader)
                plot_name = {k:v for k, v in zip(label_name, val_accuracy)}
                print('plot_name', plot_name)
                vis.plot('val_accuracy', val_accuracy)
                vis.plot('val_accuracy_gender', val_accuracy[0])

                sum_accuracy = 0.0
                for item in val_accuracy:
                    sum_accuracy += item
                print('sum_accuracy', sum_accuracy)
                vis.plot('avg_accuracy', sum_accuracy / 10.0)
        #model.save()

        # 计算验证集上的指标及可视化
        val_cm, val_accuracy = val(model, test_data_loader)
        vis.plot('val_accuracy', val_accuracy)
        sum_accuracy = 0.0
        for item in  val_accuracy:
            sum_accuracy += item
        vis.plot('avg_accuracy', sum_accuracy / 10.0)

        # 如果损失不再下降，则降低学习率
        print('loss_meter , previous_loss：',loss_meter.value()[0], previous_loss)
        if loss_meter.value()[0] > previous_loss:
            if epoch != 0:
                lr = lr * opt.gamma
                # for param_group in optimizer.param_groups:
                #     param_group['lr'] = lr

        previous_loss = loss_meter.value()[0]
        print('The %d epoch train finish.....' %epoch)


def val(model, dataloader):
    '''
    计算模型在验证集上的准确率等信息
    '''
    # 把模型设为验证模式
    model.eval()

    confusion_matrix1 = meter.ConfusionMeter(2)
    confusion_matrix2 = meter.ConfusionMeter(2)
    confusion_matrix3 = meter.ConfusionMeter(2)
    confusion_matrix4 = meter.ConfusionMeter(2)
    confusion_matrix5 = meter.ConfusionMeter(2)
    confusion_matrix6 = meter.ConfusionMeter(3)
    confusion_matrix7 = meter.ConfusionMeter(3)
    confusion_matrix8 = meter.ConfusionMeter(4)
    confusion_matrix9 = meter.ConfusionMeter(2)
    confusion_matrix10 = meter.ConfusionMeter(3)

    for ii, data in enumerate(dataloader):
        input, label = data
        val_input = Variable(input, requires_grad=False)
        val_label = Variable(label.long(), requires_grad=False)
        if opt.use_gpu:
            val_input = val_input.cuda()
            val_label = val_label.cuda()
        score = model(val_input)

        confusion_matrix1.add(score[0].data.squeeze(), label[:, 0].long())
        confusion_matrix2.add(score[1].data.squeeze(), label[:, 1].long())
        confusion_matrix3.add(score[2].data.squeeze(), label[:, 2].long())
        confusion_matrix4.add(score[3].data.squeeze(), label[:, 3].long())
        confusion_matrix5.add(score[4].data.squeeze(), label[:, 4].long())
        confusion_matrix6.add(score[5].data.squeeze(), label[:, 5].long())
        confusion_matrix7.add(score[6].data.squeeze(), label[:, 6].long())
        confusion_matrix8.add(score[7].data.squeeze(), label[:, 7].long())
        confusion_matrix9.add(score[8].data.squeeze(), label[:, 8].long())
        confusion_matrix10.add(score[9].data.squeeze(), label[:, 9].long())

    # 把模型恢复为训练模式
    model.train()
    accuracy = []

    cm_value = confusion_matrix1.value()
    accuracy1 = 100. * (cm_value[0][0] + cm_value[1][1]) / \
               (cm_value.sum())
    accuracy.append(accuracy1)

    cm_value = confusion_matrix2.value()


    accuracy2 = 100. * (cm_value[0][0] + cm_value[1][1]) / \
                (cm_value.sum())
    accuracy.append(accuracy2)
    precision = 100. * (cm_value[1][1]) / \
                (cm_value[0][1] + cm_value[1][1])
    hat_recall = 100. * (cm_value[1][1]) / \
                 (cm_value[1][0] + cm_value[1][1])
    print('hat recall:', hat_recall, cm_value[1][1], cm_value[1][0] + cm_value[1][1])
    print('hat precision:', precision)



    cm_value = confusion_matrix3.value()
    accuracy3 = 100. * (cm_value[0][0] + cm_value[1][1]) / \
                (cm_value.sum())
    accuracy.append(accuracy3)

    precision = 100. * (cm_value[1][1]) / \
                (cm_value[0][1] + cm_value[1][1])
    glass_recall = 100. * (cm_value[1][1]) / \
               (cm_value[1][0] + cm_value[1][1])
    print('glass recall:', glass_recall, cm_value[1][1], cm_value[1][0] + cm_value[1][1])
    print('glass precision:', precision)


    cm_value = confusion_matrix4.value()
    accuracy4 = 100. * (cm_value[0][0] + cm_value[1][1]) / \
                (cm_value.sum())
    accuracy.append(accuracy4)
    precision = 100. * (cm_value[1][1]) / \
                (cm_value[0][1] + cm_value[1][1])
    longcoat_recall = 100. * (cm_value[1][1]) / \
                 (cm_value[1][0] + cm_value[1][1])
    print('longcoat recall:', longcoat_recall, cm_value[1][1], cm_value[1][0] + cm_value[1][1])
    print('longcoat precision:', precision)

    cm_value = confusion_matrix5.value()
    accuracy5 = 100. * (cm_value[0][0] + cm_value[1][1]) / \
                (cm_value.sum())
    accuracy.append(accuracy5)
    precision = 100. * (cm_value[1][1]) / \
                (cm_value[0][1] + cm_value[1][1])
    boot_recall = 100. * (cm_value[1][1]) / \
                      (cm_value[1][0] + cm_value[1][1])
    print('boot recall recall:', boot_recall, cm_value[1][1], cm_value[1][0] + cm_value[1][1])
    print('boot precision:', precision)

    cm_value = confusion_matrix6.value()
    accuracy6 = 100. * (cm_value[0][0] + cm_value[1][1] + cm_value[2][2]) / \
               (cm_value.sum())
    accuracy.append(accuracy6)

    cm_value = confusion_matrix7.value()
    accuracy7 = 100. * (cm_value[0][0] + cm_value[1][1] + cm_value[2][2]) / \
               (cm_value.sum())
    accuracy.append(accuracy7)

    cm_value = confusion_matrix8.value()
    accuracy8 = 100. * (cm_value[0][0] + cm_value[1][1] + cm_value[2][2] + cm_value[3][3]) / \
               (cm_value.sum())
    accuracy.append(accuracy8)

    cm_value = confusion_matrix9.value()
    accuracy9 = 100. * (cm_value[0][0] + cm_value[1][1]) / \
               (cm_value.sum())
    accuracy.append(accuracy9)

    cm_value = confusion_matrix10.value()
    accuracy10 = 100. * (cm_value[0][0] + cm_value[1][1] + cm_value[2][2]) / \
               (cm_value.sum())
    accuracy.append(accuracy10)

    return 'confusion_matrix', accuracy


def test(**kwargs):
    opt.parse(kwargs)
    # 模型
    model = getattr(models, opt.model)().eval()
    if opt.load_model_path:
        model.load(opt.load_model_path)
    if opt.use_gpu: model.cuda()

    # 数据
    train_data = dataset.PersonData(opt.test_data_root, test=True)
    test_dataloader = dataset.PersonData(train_data, \
                                 batch_size=opt.batch_size, \
                                 shuffle=False, \
                                 num_workers=opt.num_workers)

    results = []
    for ii, (data, path) in enumerate(test_dataloader):
        input = Variable(data, volatile=True)
        if opt.use_gpu: input = input.cuda()
        score = model(input)
        probability = torch.nn.functional.softmax \
                          (score)[:, 1].data.tolist()
        batch_results = [(path_, probability_) \
                         for path_, probability_ in zip(path, probability)]
        results += batch_results
    #write_csv(results, opt.result_file)
    return results


if __name__=='__main__':
    train()