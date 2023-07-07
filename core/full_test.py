import os
import os.path as osp
import numpy as np

import torch
from torch.autograd import Variable
import torch.nn.functional as F

from core import evaluation




def full_test(net, criterion, full_testloader, full_loader, testloader, outloader,logits_min,dis_min,loss_r,unknow,epoch=None, **options):
    net.eval()

    torch.cuda.empty_cache()


    '''
    _pred = np.array([])
    _labels = np.array([])

    with torch.no_grad():

        for data, labels in testloader:
            if options['use_gpu']:
                data, labels = data.cuda(), labels.cuda()

            with torch.set_grad_enabled(False):
                data = data.squeeze(1).permute(0, 3, 1, 2)
                x, y = net(data, True)

                logits, dis, radius = criterion(x, y)
                predictions = logits.data.max(1)[1]
                # for i in range(predictions.shape[0]):
                # print('known:'+str(dis[i]))

                _pred = np.append(_pred, predictions.data.cpu().numpy())
                _labels = np.append(_labels, labels.data.cpu().numpy())


        for batch_idx, (data, labels) in enumerate(outloader):
            if options['use_gpu']:
                data, labels = data.cuda(), labels.cuda()

            with torch.set_grad_enabled(False):

                data = data.squeeze(1).permute(0, 3, 1, 2)
                x, y = net(data, True)
                # x, y = net(data, return_feature=True)
                logits, dis, radius = criterion(x, y)
                predictions = logits.data.max(1)[1]

                for i in range(predictions.shape[0]):
                    # print('unknown:'+str(dis[i]))
                    # if logits.data.max(1)[0][i].cpu()<=logits_min+radius.cpu():  # radius*10
                    # if logits.data.max(1)[0][i].cpu() <= logits_min + loss_r.cpu():
                    # if dis[i].cpu() <dis_min-radius.cpu():
                    if logits.data.max(1)[0][i].cpu() < logits_min[int(predictions[i].cpu())]:
                        predictions[i] = unknow

                _pred = np.append(_pred, predictions.data.cpu().numpy())
                _labels = np.append(_labels, labels.data.cpu().numpy())
        '''

    with torch.no_grad():

        '''
        tar = np.array([])
        pre = np.array([])
        for batch_idx, (data, labels) in enumerate(full_testloader):
            if options['use_gpu']:
                data, labels = data.cuda(), labels.cuda()

            with torch.set_grad_enabled(False):

                data = data.squeeze(1).permute(0, 3, 1, 2)
                x, y = net(data, True)
                # x, y = net(data, return_feature=True)
                logits, dis, radius = criterion(x, y)
                predictions = logits.data.max(1)[1]

                for i in range(predictions.shape[0]):
                    # print('unknown:' + str(dis[i]))
                    # if logits.data.max(1)[0][i].cpu()<=logits_min+radius.cpu():  # radius*10
                    # if logits.data.max(1)[0][i].cpu() <= logits_min + loss_r.cpu():
                    # if dis[i].cpu() <dis_min-radius.cpu():
                    if logits.data.max(1)[0][i].cpu()<logits_min[int(predictions[i].cpu())]-loss_r.cpu():
                        predictions[i] = unknow


                tar = np.append(tar, labels.data.cpu().numpy())
                pre = np.append(pre, predictions.data.cpu().numpy())
        '''

        pre_global = np.array([])

        for batch_idx, (data, labels) in enumerate(full_loader):
            if options['use_gpu']:
                data, labels = data.cuda(), labels.cuda()

            with torch.set_grad_enabled(False):

                data = data.squeeze(1).permute(0, 3, 1, 2)
                x, y = net(data, True)
                # x, y = net(data, return_feature=True)
                logits, dis, radius = criterion(x, y)
                predictions = logits.data.max(1)[1]

                for i in range(predictions.shape[0]):
                    # print('unknown:' + str(dis[i]))
                    # if logits.data.max(1)[0][i].cpu()<=logits_min+radius.cpu():  # radius*10
                    # if logits.data.max(1)[0][i].cpu() <= logits_min + loss_r.cpu():
                    # if dis[i].cpu() <dis_min-radius.cpu():
                    if logits.data.max(1)[0][i].cpu()<logits_min[int(predictions[i].cpu())]-loss_r.cpu():
                        predictions[i] = unknow

                pre_global = np.append(pre_global, predictions.data.cpu().numpy())

    return pre_global#,_pred,_labels