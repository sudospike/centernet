import sys
sys.path.remove('/home/lingfan/anaconda2/envs/centernet/lib/python3.6/site-packages')
import numpy as np
import matplotlib.pyplot as plt
N = 5
y = [20, 10, 30, 25, 15]
x = np.arange(N)

# 绘图 x x轴， height 高度, 默认：color="blue", width=0.8
p1 = plt.bar(x, height=y, width=0.5, )

# 展示图形
plt.show()

import os
import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
f = pkl.load(open("/home/remo/Desktop/remo_source/person_test/BaseLines/ghmc_log/diff.pickle"))
RAW = f[0]
GHMC = f[1]
plt.bar(np.arange(len(RAW)),RAW)
plt.show()
