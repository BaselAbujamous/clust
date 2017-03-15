import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


plt.ioff()
plt.figure(2, figsize=(5,10))
plt.subplot(3,2,2)
t = np.arange(0., 5., 0.2)
ls = plt.plot(t, t, t, t**2, t, t**3)
plt.setp(ls, color='k', linewidth=0.5)



#plt.show()

d = 5