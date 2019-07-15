import torch
import torch.nn as nn
import random
import math
import datetime
#import adabound
import os
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import utils
from torch.nn import functional as F
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

def dt():
    return datetime.datetime.now().strftime('%H:%M:%S')

def l2_norm(input,axis=1):
    norm = torch.norm(input,2,axis,True)
    output = torch.div(input, norm)
    return output

def train_softmax(model, criterion, optimizer, trainloader, use_gpu, epoch, num_classes):
    model.train()
    model.feature = False
    trainloss = 0
    for batch_idx, (data, labels) in enumerate(trainloader):
        if use_gpu:
            data, labels = data.cuda(), labels.cuda()

        _, outputs = model(data)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        trainloss += loss.item()
        
        if batch_idx % 10 == 0:
            print(dt(), 'epoch=%d batch#%d batchloss=%.4f averLoss=%.4f'
                    % (epoch, batch_idx, loss.item(), trainloss / (batch_idx + 1)))

def train_centerloss(model, criterion_xent, criterion_cent, optimizer_model, optimizer_centloss, trainloader, use_gpu, epoch, coeff=0.005):
    model.train()
    model.feature = False
    trainloss = 0
    for batch_idx, (data, labels) in enumerate(trainloader):
        if use_gpu:
            data, labels = data.cuda(), labels.cuda()

        features, outputs = model(data)

        loss_xent = criterion_xent(outputs, labels)
        loss_cent = criterion_cent(features, labels)
        loss = loss_xent + coeff*loss_cent

        optimizer_model.zero_grad()
        optimizer_centloss.zero_grad()
        loss.backward()
        optimizer_model.step()
        optimizer_centloss.step()

        trainloss += loss.item()

        if batch_idx % 10 == 0:
            print(dt(), 'epoch=%d batch#%d batchloss=%.4f averLoss=%.4f, loss_xent=%.4f, loss_centws=%.4f'
                    % (epoch, batch_idx, loss.item(), trainloss / (batch_idx + 1), loss_xent.item(), loss_cent.item()))
            
def train_arc(model, criterion, criterion_arc, optimizer, trainloader, use_gpu, epoch, num_classes):
    model.train()
    model.feature = False
    trainloss = 0
    for batch_idx, (data, labels) in enumerate(trainloader):
        if use_gpu:
            data, labels = data.cuda(), labels.cuda()

        features, outputs = model(data)
        loss = criterion(criterion_arc(features, labels), labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        trainloss += loss.item()
        
        if batch_idx % 10 == 0:
            print(dt(), 'epoch=%d batch#%d batchloss=%.4f averLoss=%.4f'
                    % (epoch, batch_idx, loss.item(), trainloss / (batch_idx + 1)))   

"""
center loss, ECCV 2016, the full loss should be centerloss + softmaxloss    
"""     
class CenterLoss(nn.Module):
    def __init__(self, num_classes=10574, feat_dim=64, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        batch_centers = self.centers[labels, :]
        diff = (batch_centers - x)
        loss = diff.pow(2).sum(1).clamp(min=1e-12, max=1e+12).mean()

        return loss
    
"""
Arcface ArcFace: Additive Angular Margin Loss for Deep Face Recognition
"""
class Arcface(nn.Module):
    # implementation of additive margin softmax loss in https://arxiv.org/abs/1801.05599    
    def __init__(self, embedding_size=512, classnum=28,  s=64., m=0.5):
        super(Arcface, self).__init__()
        self.classnum = classnum
        self.kernel = nn.Parameter(torch.Tensor(embedding_size,classnum))
        # initial kernel
        self.kernel.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        self.m = m # the margin value, default is 0.5
        self.s = s # scalar value default is 64, see normface https://arxiv.org/abs/1704.06369
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.mm = self.sin_m * m  # issue 1
        self.threshold = math.cos(math.pi - m)
    def forward(self, embbedings, label):
        # weights norm
        nB = len(embbedings)
        kernel_norm = l2_norm(self.kernel,axis=0)
        # cos(theta+m)
        cos_theta = torch.mm(embbedings,kernel_norm)
#         output = torch.mm(embbedings,kernel_norm)
        cos_theta = cos_theta.clamp(-1,1) # for numerical stability
        cos_theta_2 = torch.pow(cos_theta, 2)
        sin_theta_2 = 1 - cos_theta_2
        sin_theta = torch.sqrt(sin_theta_2)
        cos_theta_m = (cos_theta * self.cos_m - sin_theta * self.sin_m)
        # this condition controls the theta+m should in range [0, pi]
        #      0<=theta+m<=pi
        #     -m<=theta<=pi-m
        cond_v = cos_theta - self.threshold
        cond_mask = cond_v <= 0
        keep_val = (cos_theta - self.mm) # when theta not in [0,pi], use cosface instead
        cos_theta_m[cond_mask] = keep_val[cond_mask]
        output = cos_theta * 1.0 # a little bit hacky way to prevent in_place operation on cos_theta
        idx_ = torch.arange(0, nB, dtype=torch.long)
        output[idx_, label] = cos_theta_m[idx_, label]
        output *= self.s # scale up in order to make softmax work, first introduced in normface
        return output

"""
this kernel does not work
"""
def cosine_kernel(X,Y):
  return np.dot(X, Y.T)/(np.linalg.norm(X)*np.linalg.norm(Y))

def save_model(model, filename):
    state = model.state_dict()
    for key in state:
        state[key] = state[key].clone().cpu()
    torch.save(state, filename)
    
def KFold(n=1200, n_folds=5, shuffle=False):
    folds = []
    base = list(range(n))
    for i in range(n_folds):
        test = base[i*n//n_folds:(i+1)*n//n_folds]
        train = list(set(base)-set(test))
        folds.append([train,test])
    return folds


def eval_acc(threshold, diff):
    y_true = []
    y_predict = []
    for d in diff:
        same = 1 if float(d[0]) > threshold else 0
        y_predict.append(same)
        y_true.append(int(d[1]))
    y_true = np.array(y_true)
    y_predict = np.array(y_predict)
    accuracy = 1.0*np.count_nonzero(y_true==y_predict)/len(y_true)
    return accuracy

"""
not recommend, only used for NN classification
"""
def evaluate_pure_pytorch(model, testloader, use_gpu, epoch):
    model.eval()
    model.feature = False

    train_features, train_labels = [], []
    test_features, test_labels = [], []
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(testloader):
            if use_gpu:
                data, labels = data.cuda(), labels

            _, output = model(data)
            _, predicted = torch.max(output.data.cpu(), 1)

            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = float(correct)/total
    print('epoch= %d, svm classification accuracy: %.4f' % (epoch, acc))
    return acc

"""
cosine distance verification test
"""
def find_best_threshold(thresholds, predicts):
    best_threshold = best_acc = 0
    for threshold in thresholds:
        accuracy = eval_acc(threshold, predicts)
        if accuracy >= best_acc:
            best_acc = accuracy
            best_threshold = threshold
    return best_threshold

def evaluate(model, testloader, use_gpu, epoch):
    model.eval()
    #ignore the classifier
    model.feature = True
    predicts = []

    with torch.no_grad():
        for batch_idx, (img1, img2, sameflag) in enumerate(testloader):
            if use_gpu:
                img1 = img1.float().cuda()
                img2 = img2.float().cuda()
            else:
                img1 = img1.float()
                img2 = img2.float()
            f1 = model(img1)
            f2 = model(img2)

            for i in range(len(f1)):
                cosdistance = f1[i].view(1, -1).mm(f2[i].view(-1, 1)) / (f1[i].norm() * f2[i].norm() + 1e-5)
                # predicts.append('{}\t{}\n'.format(cosdistance, sameflag[i]))
                if use_gpu:
                    predicts.append((cosdistance.cpu().item(), sameflag[i].item()))
                else:
                    predicts.append((cosdistance.item(), sameflag[i].item()))

    accuracy = []
    thd = []
    folds = KFold(n=4272, n_folds=4, shuffle=False)
    thresholds = np.arange(-1.0, 1.0, 0.005)  # cosine distance
    for idx, (train, test) in enumerate(folds):
        best_thresh = find_best_threshold(thresholds, [predicts[i] for i in train])
        accuracy.append(eval_acc(best_thresh, [predicts[i] for i in test]))
        thd.append(best_thresh)
    print('Epoch={} acc={:.4f} std={:.4f} thd={:.4f}'.format(epoch, np.mean(accuracy), np.std(accuracy), np.mean(thd)))
    return np.mean(accuracy), np.std(accuracy), np.mean(thd)


def evaluate_identification(model, trainloader, testloader, use_gpu, epoch):
    model.eval()
    model.feature = True

    train_features, train_labels = [], []
    test_features, test_labels = [], []

    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(trainloader):
            if use_gpu:
                data, labels = data.cuda(), labels.cuda()

            features = model(data)

            if use_gpu:
                train_features.append(features.data.cpu().numpy())
                train_labels.append(labels.data.cpu().numpy())
            else:
                train_features.append(features.data.numpy())
                train_labels.append(labels.data.numpy())

        train_features = np.concatenate(train_features, 0)
        train_labels = np.concatenate(train_labels, 0)

    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(testloader):
            if use_gpu:
                data, labels = data.cuda(), labels.cuda()

            features = model(data)

            if use_gpu:
                test_features.append(features.data.cpu().numpy())
                test_labels.append(labels.data.cpu().numpy())
            else:
                test_features.append(features.data.numpy())
                test_labels.append(labels.data.numpy())

        test_features = np.concatenate(test_features, 0)
        test_labels = np.concatenate(test_labels, 0)
    #svm_model = XGBClassifier()
    #svm_model = LinearSVC(C=100, random_state=42, max_iter=5000)
    svm_model = LinearSVC(C=1, random_state=42)
    svm_model.fit(train_features, train_labels)
    labels_pred = svm_model.predict(test_features)
    #scores = svm_model.predict_proba(test_features)
    #scores_max = scores[range(len(scores)), scores.argmax(axis=1)]
    acc_test = np.sum(np.array(labels_pred) == np.array(test_labels)) / len(test_labels)

    print('epoch= %d, svm classification accuracy: %.4f' % (epoch, acc_test))
    return acc_test