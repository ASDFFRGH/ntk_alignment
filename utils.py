import torch
import torchvision
from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader 
from torch import nn
import torch.optim as optim
import numpy as np
import pandas as pd
from models import *

def init_teacher(model):
  with torch.no_grad():
       model.fc1.weight = nn.Parameter(torch.Tensor([[ 1,  2],
                                                     [ 3,  4],
                                                     [ 5,  6],
                                                     [ 1,  4],
                                                     [ 2,  4]]))     
       model.fc1.bias = nn.Parameter(torch.Tensor([ 5,  4,  3,  2,  1]))
       model.fc2.weight = nn.Parameter(torch.Tensor([[ 4,  5,  1,  2,  2]]))
       model.fc2.bias = nn.Parameter(torch.Tensor([ 0.4365]))

def sythetic_data(num_examples, std, dim):
  model = teacher_model()
  init_teacher(model)
  with torch.no_grad():
    X = torch.normal(0, 3, size=(num_examples, dim))
    t = model(X)
    t += torch.normal(0, std, t.shape)
  return X, t.reshape((-1, 1))

def get_data_loader(X, t, batch_size):
  dataset = []
  for input, output in zip(X, t):
    dataset.append((input, output))

  data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
  
  return data_loader

def param_xavier_init(model):
    for name, param in model.named_parameters():
        if 'weight' in name:
          nn.init.xavier_normal_(param)
          print("do xavier init weight")
        if 'bias' in name:
          nn.init.constant_(param, val=0)
          print("do init bias")

def param_init(model, std):
  for name, param in model.named_parameters():
    if 'weight' in name:
      nn.init.normal_(param, mean=0, std=std)
      print("do init w")
    if 'bias' in name:
      nn.init.constant_(param, val=0)
      print("do init b")

def decide_optimizer(model, opt, learning_rate):
  if(opt=='SGD'):
    optimizer = optim.SGD(model.parameters(), lr = learning_rate, weight_decay = 0) 
  elif(opt=='Adam'):
    optimizer = optim.Adam(model.parameters(), lr = learning_rate, weight_decay = 0)
  elif(opt=='RMSprop'):
    optimizer = optim.RMSprop(model.parameters(), lr = learning_rate, weight_decay = 0) 

  return optimizer

def ntk(model, inp, device):
  """Calculate the neural tangent kernel of the model on the inputs.
  Returns the gradient feature map along with the tangent kernel.
  """
  #inp = torch.tensor(inp, requires_grad=True)
  out = model(inp).to(device)

  #p_vec = nn.utils.parameters_to_vector(model.parameters())
  n, outdim = out.shape
  assert outdim == 1, "cant handle output dim higher than 1 for now"

  features2 = [ torch.zeros( (n,) + tuple(param.shape) ).to(device)  for param in model.parameters() ]
  tk2 = torch.zeros(n,n).to(device)

  for i in range(n):
      model.zero_grad()
      out[i].backward(retain_graph=True)
      for j, param in enumerate(model.parameters()):
          features2[j][i,:] = param.grad
  v_dim = 0
  for feature in features2:
      v_feature = feature.view(n,-1)
      v_dim += v_feature.shape[1]      
  for feature in features2:
      v_feature = feature.view(n,-1)
      tk2 += v_feature @ v_feature.t() / v_dim

  return features2, tk2

def get_max(lam):
  eigvals,_ = torch.chunk(lam, 2, dim=1)
  #eigvals,_ = torch.sort(eigvals, descending=True, dim=0)

  return eigvals[0]  

def erank(lam):
  eigvals,_ = torch.chunk(lam, 2, dim=1)
  #eigvals,_ = torch.sort(eigvals, descending=True, dim=0)
  num = eigvals.shape[0]
  eigvals_sum = torch.sum(eigvals)
  H = 0
  for i in range(num):
    if i == 0:
      mu = eigvals[i] / eigvals_sum
    else:
      mu = torch.cat((mu, eigvals[i] / eigvals_sum), dim=0)
  for i in range(num):
      log_mu_i = torch.log(mu[i])
      if torch.isnan(log_mu_i):
        log_mu_i = 0
      H -= mu[i] * log_mu_i
  
  return torch.exp(H)

def trace_ratio(lam, k):
  eigvals,_ = torch.chunk(lam, 2, dim=1)
  #eigvals,_ = torch.sort(eigvals, descending=True, dim=0)
  num = eigvals.shape[0]
  eigvals_sum = torch.sum(eigvals)
  sum_k = 0
  for i in range(k):
    sum_k += eigvals[i]
  trace_ratio = sum_k / eigvals_sum

  return trace_ratio

def kernel_size(tk):
  return torch.sqrt(torch.trace(tk.t()@tk))

"""
def alignment(eigvecs, t, k):
  cos = 0
  for i in range(k):
    eigvec_i = eigvecs[i].to(device)
    cos += (torch.t(eigvec_i) @ t.to(device).float()) ** 2
    #cos = (torch.t(eigvec_i) @ t.to(device).float()) ** 2
  norm = torch.linalg.norm(t.to(device).float()) ** 2

  return cos / norm

def alignment(eigvecs, t, k):
  v = eigvecs[k]
  a = torch.t(v) @ t
  norm = torch.linalg.norm(v) * torch.linalg.norm(t)

  return a/norm"""

def similarity(tk, t):
    t = t.float()
    a = t.t() @ tk @ t
    norm = torch.sqrt(torch.trace(tk.t()@tk)) * torch.linalg.norm(t)**2

    return a/norm      

#未完成
def kernel_change_rate(g, G):
    a = g.t() - G
    a = torch.sqrt(torch.trace(a.t()@a))
    norm = torch.sqrt(torch.trace(g.t()@g)) * torch.sqrt(torch.trace(G.t()@G))
    return a / norm

def save_data(opt, data):
    print('データ保存前')
    print(data.gd_iter)
    pd.to_pickle(data, "data/"+opt+"_inds.pkl")
    print('データ保存後')
    print(data.gd_iter)
    print("saved..."+opt)


def load_data(opt, root):
    path=root+ opt +'_inds.pkl'
    ind = pd.read_pickle(path)
    return ind

class data():
  gd_iter = None
  gd_iter2 = None
  erank = None
  alignment = None
  all_loss = None
  test_error = None
  k_size = None
  kcr = None
  tr5 = None
  tr10 = None
  tr30 = None
  std = None
  lr = None
  initialized = None
  def __init__(self):
      self.gd_iter = []
      self.gd_iter2 = []
      self.erank = []
      self.alignment = []
      self.all_loss = []
      self.test_error = []
      self.k_size = []
      self.kcr = []
      self.tr5 = []
      self.tr10 = []
      self.tr30 = []
      self.std = 0
      self.lr = 0
      self.initialized = 'none'


      