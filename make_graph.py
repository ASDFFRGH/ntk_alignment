import numpy as np
import matplotlib.pyplot as plt

def make_graph(p, t, train_loss, test_error, indicator, name, opt):

  fig = plt.figure()
  bx = fig.add_subplot(211)
  ax = fig.add_subplot(212)
  ax.plot(p, indicator)
  ax.set_xlabel('iterations')
  ax.set_ylabel(name)
  bx.set_title(opt)
  bx.plot(t, train_loss, label = 'train loss')
  bx.plot(t, test_error, label = 'test error')
  bx.set_ylabel('loss')
  bx.set_ylim(0, 1)
  fig.legend() 
  plt.grid()
  plt.show()

def make_compare_graph1(p, ind1, ind2, ind3, name, label1, label2, label3):

  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.plot(p, ind1, label = label1)
  ax.plot(p, ind2, label = label2)
  ax.plot(p, ind3, label = label3)
  ax.set_xlabel('iterations')
  ax.set_ylabel(name)
  fig.legend() 
  plt.grid()
  plt.show()


def more_detail(p, ind1, ind2, ind3, name, label1, label2, label3, a, b):
  
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.plot(p, ind1, label = label1)
  ax.plot(p, ind2, label = label2)
  ax.plot(p, ind3, label = label3)
  ax.set_xlabel('iterations')
  ax.set_ylabel(name)
  plt.ylim(a, b)
  fig.legend() 
  plt.grid()
  plt.show()

def make_compare_graph2(p, train_loss, test_error, ind1, ind2, ind3, name, opt, label1, label2, label3):
  
  fig = plt.figure()
  bx = fig.add_subplot(211)
  ax = fig.add_subplot(212)
  ax.plot(p, ind1, label = label1)
  ax.plot(p, ind2, label = label2)
  ax.plot(p, ind3, label = label3)
  ax.set_xlabel('iterations')
  ax.set_ylabel(name)
  bx.set_title(opt)
  bx.plot(p, train_loss, label = 'train loss')
  bx.plot(p, test_error, label = 'test error')
  bx.set_ylabel('loss')
  fig.legend() 
  plt.grid()
  plt.show()