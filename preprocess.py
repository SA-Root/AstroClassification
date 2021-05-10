import numpy as np
from sklearn import model_selection

tx = np.load('data/first_train_data_x.npy',
             allow_pickle=True).astype(np.float32)
ty = np.load('data/first_train_data_y.npy', allow_pickle=True)

xt, xv, yt, yv = model_selection.train_test_split(tx, ty, test_size=0.2)

res = []
resy = []
res2 = []
resy2 = []
res3 = []
resy3 = []

for i in range(0, 7000):
    if yt[i] == 'qso':
        for j in range(0, 200):
            res.append(xt[i])
            resy.append(yt[i])
    elif yt[i] == 'unknown':
        for j in range(0, 10):
            res2.append(xt[i])
            resy2.append(yt[i])
    elif yt[i] == 'galaxy':
        for j in range(0, 60):
            res3.append(xt[i])
            resy3.append(yt[i])

xt = np.append(xt, res, axis=0)
xt = np.append(xt, res2, axis=0)
xt = np.append(xt, res3, axis=0)
ntx = []
for i in xt:
    ntx.append(i.reshape((2600, 1, 1)))
ntx = np.array(ntx, dtype=np.float32)

nvx = []
for i in xv:
    nvx.append(i.reshape((2600, 1, 1)))
nvx = np.array(nvx, dtype=np.float32)

yt = np.append(yt, resy)
yt = np.append(yt, resy2)
yt = np.append(yt, resy3)
nty = []

for i in yt:
    if i == 'star':
        nty.append([1, 0, 0, 0])
    elif i == 'unknown':
        nty.append([0, 1, 0, 0])
    elif i == 'galaxy':
        nty.append([0, 0, 1, 0])
    else:
        nty.append([0, 0, 0, 1])
nty = np.array(nty, dtype=np.float32)

nvy = []

for i in yv:
    if i == 'star':
        nvy.append([1, 0, 0, 0])
    elif i == 'unknown':
        nvy.append([0, 1, 0, 0])
    elif i == 'galaxy':
        nvy.append([0, 0, 1, 0])
    else:
        nvy.append([0, 0, 0, 1])
nvy = np.array(nvy, dtype=np.float32)


np.save('data/xt.npy', ntx)
np.save('data/xv.npy', nvx)
np.save('data/yt.npy', nty)
np.save('data/yv.npy', nvy)
