import numpy as np
from submit import my_map
from submit import my_fit
# from sampleSOl import my_map
# from sampleSOl import my_fit
import time as tm

Z_trn = np.loadtxt( "secret_trn.txt")
Z_tst = np.loadtxt( "secret_tst.txt" )

n_trials = 5

d_size = 0
t_train = 0
t_map = 0
acc0 = 0
acc1 = 0

for t in range(n_trials):
    tic = tm.perf_counter()
    w0, b0, w1, b1 = my_fit(Z_trn[:, :-2], Z_trn[:, -2], Z_trn[:,-1])
    toc = tm.perf_counter()
    
    t_train += toc - tic
    w0 = w0.reshape(-1)
    w1 = w1.reshape(-1)
    
    d_size += max(w0.shape[0], w1.shape[0])
    
    tic = tm.perf_counter()
    feat = my_map(Z_tst[:, :-2])
    # print(feat.shape)
    toc = tm.perf_counter()
    t_map += toc - tic
    
    scores0 = feat.dot(w0) + b0
    scores1 = feat.dot(w1) + b1
    
    pred0 = np.zeros_like(scores0)
    pred0[scores0 > 0] = 1
    pred1 = np.zeros_like(scores1)
    pred1[scores1 > 0] = 1
    
    acc0 += np.average(Z_tst[:, -2] == pred0)
    acc1 += np.average(Z_tst[:, -1] == pred1)
d_size /= n_trials
t_train /= n_trials
t_map /= n_trials
acc0 /= n_trials
acc1 /= n_trials
print( f"{d_size},{t_train},{t_map},{acc0},{acc1}" )