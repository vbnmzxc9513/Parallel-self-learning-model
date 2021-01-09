import time
start = time.time()
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import os
import pandas as pd
from tqdm import tqdm, tqdm_notebook 
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split

class SVM_cuda(nn.Module):
    #列出需要哪些層
    def __init__(self, feature_num, cls_num):
        super(SVM_cuda, self).__init__()
        self.fc = nn.Linear(feature_num, cls_num)     
    #列出forward的路徑，將init列出的層代入
    def forward(self, x):
        out = self.fc(x) 
        return out
    
class HingeLoss(torch.nn.Module):

    def __init__(self):
        super(HingeLoss, self).__init__()
        self.relu = nn.ReLU()

    def forward(self, output, target):
        all_ones = torch.ones_like(target)
        labels = 2 * target - all_ones
        losses = all_ones - torch.mul(output, labels)

        return torch.norm(self.relu(losses))
    
    
def SVM_train(training_data, val_data, test_data, config):
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
#     device = 'cpu'
    training_data = training_data.to(device)
    val_data = val_data.to(device)
    test_data = test_data.to(device)

    svm = SVM_cuda(config.feature_num, config.cls_num).to(device)
    cu_lr=0.001
    optimizer = optim.Adam(svm.parameters(), lr=cu_lr)
    criterion = torch.nn.CrossEntropyLoss()
    # criterion = HingeLoss()

    best_acc = 0
    best_model = None
#     early_stop = 50

    
    
    for epoch in tqdm(range(config.epoch)):
        training_data = training_data[torch.randperm(training_data.size()[0])].float()
        val_data = val_data[torch.randperm(val_data.size()[0])].float()
        test_data = test_data[torch.randperm(test_data.size()[0])].float()
        
        sum_loss = 0
        train_total = 0
        train_true = 0
        val_total = 0
        val_true = 0
        
        if epoch+1 % 1000 == 0:
            cu_lr = cu_lr / 5
            optimizer = optim.Adam(svm.parameters(), lr=cu_lr)

        ########################                    
        # train the model      #
        ########################
        svm.train()
        for i in range(0, len(training_data), config.batch_size):
            x = training_data[i:i+config.batch_size, :-1]
            y = training_data[i:i+config.batch_size, -1].long()

            optimizer.zero_grad()
            
            output = svm(x)
            prob, pred = torch.relu(output).max(1)

            train_true += torch.sum(pred == y).item()
            # loss = criterion(prob, y)
            loss = criterion(output, y)
            # loss = torch.mean(torch.clamp(1 - y * prob, min=0))
            # loss += config.c / 2.0
            
            loss.backward()
            optimizer.step()
            
            sum_loss += float(loss)
            train_total += len(y)
        # print("train: epoch: {:4d}, loss: {:.3f}, accuracy: {}".format(epoch, sum_loss / train_total, train_true/ train_total))

        ########################
        # validate the model   #
        ########################
        svm.eval()
        for i in range(0, len(val_data), config.batch_size):
            x = val_data[i:i+config.batch_size, :-1]
            y = val_data[i:i+config.batch_size, -1]

            optimizer.zero_grad()
            
            with torch.no_grad():
                prob, pred = torch.relu(svm(x)).max(1)

            val_true += torch.sum(pred == y).item()
            val_total += len(y)
#         print("validation: epoch: {:4d}, loss: {:.3f}, accuracy: {}".format(epoch, sum_loss / val_total, val_true/ val_total))
        if best_acc <= val_true/ val_total:
            best_acc = val_true/ val_total
            best_model = copy.deepcopy(svm)
        

#     evaluation(best_model, test_data)
    return best_model


def evaluation(svm, test_data, config):
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
#     device = 'cpu'
    test_data = test_data.float().to(device)
    svm = svm.to(device)
    svm.eval()
    test_true = [0 for i in range(config.cls_num)]
    test_total = [0 for i in range(config.cls_num)]

    for i in range(0, len(test_data), config.batch_size):
            x = test_data[i:i+config.batch_size, :-1]
            y = test_data[i:i+config.batch_size, -1]
            
            with torch.no_grad():
                prob, pred = torch.relu(svm(x)).max(1)
            
            for i in range(config.cls_num):
                test_true[i] += torch.sum((pred == i) * (i == y)).item()
                test_total[i] += sum(y == i).item()
                #print(y)
                #print(i)
                #print('------------')
                #print(i, (pred == y), sum(y == i))
            
            #for i in range(7):
             #   print( test_true[i], test_total[i])
    
    print('Apple_pie accuracy %f\n' % (test_true[0] /test_total[0]))
    print('Chocolate_cake accuracy: %f\n' % (test_true[1] /test_total[1]))
    print('Donuts accuracy: %f\n' % (test_true[2] /test_total[2]))
    print('Hamburger accuracy: %f\n' % (test_true[3] /test_total[3]))
    print('Hot_dog accuracy: %f\n' % (test_true[4] /test_total[4]))
    print('Ice_cream accuracy: %f\n' % (test_true[5] /test_total[5]))
    print('Pizza accuracy: %f\n' % (test_true[6] /test_total[6]))
    
    
class Config():
    def __init__(self):
        self.feature_num = 0
        self.cls_num = 7
        # self.c = 1
        self.batch_size = 2500
        self.epoch = 6000
        
        
config = Config()

train_path = os.path.join('.', 'Train_food7.csv')
train_pd = pd.read_pickle('train_food7.pkl')
test_pd = pd.read_pickle('test_food7.pkl')

config.feature_num = train_pd.shape[1]-1
training_data, test_data = torch.tensor(train_pd.values), torch.tensor(test_pd.values)

svm = SVM_train(training_data, test_data, test_data, config)
evaluation(svm, test_data, config)
