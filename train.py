import torch
import torchvision
from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader 
from torch import nn
import torch.optim as optim
import numpy as np
import pandas as pd
from models import *
from utils import *
from make_graph import *

def train_and_ntk(model, optimizer, loss_fn, iteration):

  ind = data()
  count = 0
  #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.5)
  #トレーニング前のntkの計算
  m = torch.load(model_path)
  ind.gd_iter.append(count)
  _, grammatrix = ntk(m, X_train.to(device), device)
  eigvals, v = torch.eig(grammatrix, eigenvectors=True)
  print('get_ntk: {}' .format(len(ind.gd_iter)) )
  #ntkの固有値固有ベクトルから各指標を計算
  ind.k_size.append(kernel_size(grammatrix).item())
  ind.erank.append(erank(eigvals).tolist())
  ind.tr5.append(trace_ratio(eigvals, 5).item())
  ind.tr10.append(trace_ratio(eigvals, 10).item())
  ind.tr30.append(trace_ratio(eigvals, 30).item())
  ind.alignment.append(similarity(grammatrix, t_train).item())
  #print(len(ind.gd_iter), len(ind.alignment))  

  old_g = grammatrix


  for epoch in range(iteration+1):
    train_loss = []
    test_loss = []
    total_train = 0
    total_test= 0

    model.train()
    for X, t in train_loader:
      optimizer.zero_grad()
      y = model(X.to(device))
      t = t.view(-1, 1)
      loss = loss_fn(y.to(device), t.to(device).float())
      loss.backward()
      optimizer.step()
      #scheduler.step()

      train_loss.append(loss.tolist())

      count += 1
      if count <= 100 or count%10 == 0:
      #if count <= 100:
        ind.gd_iter.append(count)
        _, grammatrix = ntk(model, X_train.to(device), device)
        eigvals, v = torch.eig(grammatrix, eigenvectors=True)
        print('get_ntk: {}' .format(len(ind.gd_iter)) )
        #ntkの固有値固有ベクトルから各指標を計算
        ind.k_size.append(kernel_size(grammatrix).item())
        ind.erank.append(erank(eigvals).tolist())
        ind.tr5.append(trace_ratio(eigvals, 5).item())
        ind.tr10.append(trace_ratio(eigvals, 10).item())
        ind.tr30.append(trace_ratio(eigvals, 30).item())
        ind.alignment.append(similarity(grammatrix, t_train).item())
        ind.kcr.append(kernel_change_rate(old_g, grammatrix).item())
        print(len(ind.gd_iter), len(ind.alignment))
        old_g = grammatrix


    model.eval()
    for X, t in test_loader:
      y = model(X.to(device))
      t = t.view(-1, 1)
      loss = loss_fn(y.to(device), t.to(device).float())

      test_loss.append(loss.tolist())
  
    #１epoch終了ごとにグラムマトリクスを求めるならここs  
    ind.gd_iter2.append(count)
    ind.all_loss.append(np.mean(train_loss))
    ind.test_error.append(np.mean(test_loss))

    print('EPOCH: {}, train[loss: {:.10f}], test[loss: {:.10f}]' 
        .format(epoch, np.mean(train_loss), np.mean(test_loss)))
  
  print("Calculated NTK times: {}" .format(count/10))

  return ind

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

std = 0.01
dim = 2
#X_train, t_train = sythetic_data(100, std, dim)
#X_test, t_test = sythetic_data(1000,std, dim)
if std == 0:
  path1 = 'synthetic_data/train_std=0'
  path2 = 'synthetic_data/test_std=0'
elif std == 0.01:  
  path1 = 'synthetic_data/train_std=0.01'
  path2 = 'synthetic_data/test_std=0.01'
elif std == 0.1:
  path1 = 'synthetic_data/train_std=0.1'
  path2 = 'synthetic_data/test_std=0.1'

'''
pd.to_pickle(X_train, path1 + 'X')
pd.to_pickle(t_train, path1 + 't')
pd.to_pickle(X_test, path2 + 'X')
pd.to_pickle(t_test, path2 + 't')
'''
X_train = pd.read_pickle(path1 + 'X')
t_train = pd.read_pickle(path1 + 't')
X_test = pd.read_pickle(path2 + 'X')
t_test = pd.read_pickle(path2 + 't')

batch_size = 64
train_loader = get_data_loader(X_train, t_train, batch_size)
test_loader = get_data_loader(X_test, t_test, batch_size)



opt = 'SGD'
opt2 = 'Adam'
opt3 = 'RMSprop'
iteration = 3000

#model = Net().to(device)
#p_std = 0.0001
#param_xavier_init(model)
#param_init(model, p_std)
#model_path = 'initialized_model_std1e-4.pth'
model_path = 'initialized_model_xavier.pth'
#torch.save(model, model_path)

#SGD
model = torch.load(model_path)
loss_fn = nn.MSELoss()
lr1 = 3.0
optimizer = decide_optimizer(model, opt, lr1)
#gd_iter, gd_iter2, loss, test_error, eranks, tr5, tr10, tr30, alignment, k_size = train_and_ntk(model, optimizer, loss_fn, iteration)
#save_data(opt, gd_iter, gd_iter2, loss, test_error, eranks, tr5, tr10, tr30, alignment, k_size)
data1 = train_and_ntk(model, optimizer, loss_fn, iteration)
data1.std = std
data1.initialized = model_path
data1.lr = lr1
save_data(opt, data1)

#Adam
model = torch.load(model_path)
opt2 = "Adam"
lr2 = 0.1
optimizer = decide_optimizer(model, opt2, lr2)

data2 = train_and_ntk(model, optimizer, loss_fn, iteration)
data2.std = std
data2.lr = lr2
#gd_iter, gd_iter2, loss2, test_error2, eranks2, tr5_2, tr10_2, tr30_2, alignment2, kernel_size2 = train_and_ntk(model, optimizer, loss_fn, iteration)
#save_data(opt2, gd_iter, gd_iter2, loss2, test_error2, eranks2, tr5_2, tr10_2, tr30_2, alignment2,kernel_size2)
save_data(opt2, data2)

#RMSprop
model = torch.load(model_path)
opt3 = "RMSprop"
lr3 = 0.01
optimizer = decide_optimizer(model, opt3, lr3)

#gd_iter, gd_iter2, loss3, test_error3, eranks3, tr5_3, tr10_3, tr30_3, alignment3, kernel_size3 = train_and_ntk(model, optimizer, loss_fn, iteration)
#save_data(opt3, gd_iter, gd_iter2, loss3, test_error3, eranks3, tr5_3, tr10_3, tr30_3, alignment3, kernel_size3)
data3 = train_and_ntk(model, optimizer, loss_fn, iteration)
data3.std = std
data3.lr = lr3
save_data(opt3, data3)

print('std=',std)