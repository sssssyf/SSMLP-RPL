import os
import os.path as osp
import numpy as np

import torch
from torch.autograd import Variable
import torch.nn.functional as F

from core import evaluation

def test(net, criterion,full_testloader, testloader, outloader,logits_min,dis_min,loss_r,unknow, epoch=None, **options):
    net.eval()

    correct, total = 0, 0

    correct_u, total_u = 0, 0

    correct_all, total_all = 0, 0

    torch.cuda.empty_cache()

    _pred_k, _pred_u, _pred_all, _labels = [],[],[], []

    with torch.no_grad():
        for data, labels in testloader:
            if options['use_gpu']:
                data, labels = data.cuda(), labels.cuda()
            
            with torch.set_grad_enabled(False):
                data = data.squeeze(1).permute(0, 3, 1, 2)
                x, y = net(data, True)

                logits, dis, radius = criterion(x, y)
                predictions = logits.data.max(1)[1]
                #for i in range(predictions.shape[0]):
                    #print('known:'+str(dis[i]))
                total += labels.size(0)
                correct += (predictions == labels.data).sum()
            
                _pred_k.append(logits.data.cpu().numpy())
                _labels.append(labels.data.cpu().numpy())

        for batch_idx, (data, labels) in enumerate(outloader):
            if options['use_gpu']:
                data, labels = data.cuda(), labels.cuda()
            
            with torch.set_grad_enabled(False):

                data = data.squeeze(1).permute(0, 3, 1, 2)
                x, y = net(data, True)
                # x, y = net(data, return_feature=True)
                logits, dis, radius = criterion(x, y)
                predictions = logits.data.max(1)[1]

                total_u += labels.size(0)

                for i in range(predictions.shape[0]):
                    #print('unknown:'+str(dis[i]))
                    #if logits.data.max(1)[0][i].cpu()<=logits_min+radius.cpu():  # radius*10
                    #if logits.data.max(1)[0][i].cpu() <= logits_min + loss_r.cpu():
                    #if dis[i].cpu() <dis_min-radius.cpu():
                    if logits.data.max(1)[0][i].cpu()<logits_min[int(predictions[i].cpu())]-loss_r.cpu():
                        predictions[i] = unknow
                correct_u += (predictions == labels.data).sum()

                _pred_u.append(logits.data.cpu().numpy())

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

                total_all += labels.size(0)

                for i in range(predictions.shape[0]):
                    #print('unknown:' + str(dis[i]))
                    # if logits.data.max(1)[0][i].cpu()<=logits_min+radius.cpu():  # radius*10
                    # if logits.data.max(1)[0][i].cpu() <= logits_min + loss_r.cpu():
                    # if dis[i].cpu() <dis_min-radius.cpu():
                    #if logits.data.max(1)[0][i].cpu()<logits_min[int(predictions[i].cpu())]-loss_r.cpu():
                    if logits.data.max(1)[0][i].cpu() < logits_min[int(predictions[i].cpu())] - loss_r.cpu():
                        predictions[i] = unknow
                correct_all += (predictions == labels.data).sum()

                tar = np.append(tar, labels.data.cpu().numpy())
                pre = np.append(pre, predictions.data.cpu().numpy())

                #_pred_u.append(logits.data.cpu().numpy())

    # Accuracy
    acc = float(correct) * 100. / float(total)
    print('Acc: {:.5f}'.format(acc))

    acc_u = float(correct_u) * 100. / float(total_u)
    print('Acc_u: {:.5f}'.format(acc_u))

    acc_all = float(correct_all) * 100. / float(total_all)
    print('Acc_all: {:.5f}'.format(acc_all))

    _pred_k = np.concatenate(_pred_k, 0)
    _pred_u = np.concatenate(_pred_u, 0)
    _labels = np.concatenate(_labels, 0)


    # Out-of-Distribution detction evaluation
    x1, x2 = np.max(_pred_k, axis=1), np.max(_pred_u, axis=1)
    results = evaluation.metric_ood(x1, x2)['Bas']
    
    # OSCR
    _oscr_socre = evaluation.compute_oscr(_pred_k, _pred_u, _labels)

    results['ACC'] = acc
    results['OSCR'] = _oscr_socre * 100.

    return results,pre,tar


