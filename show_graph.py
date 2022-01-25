import numpy as np
import matplotlib.pyplot as plt
from utils import *
from make_graph import *

opt = 'SGD'
opt2 = 'Adam'
opt3= 'RMSprop'

gd_iter, gd_iter2, erank, alignment, loss, test_error, k_size, tr5, tr10, tr30 = load_data(opt)
#_, _, erank2, alignment2, loss2, test_error2, k_size2, tr5_2, tr10_2, tr30_2 =load_data(opt2)
#_, _, erank3, alignment3, loss3, test_error3, k_size3, tr5_3, tr10_3, tr30_3 = load_data(opt3)

make_graph(gd_iter, gd_iter2, loss, test_error, alignment, 'alignment', opt)