import math
import numpy
import matplotlib.pyplot as plt
x= numpy.linspace(-2,2,1000)
fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
plt.plot(x, 0.5*(1-x**2)**2, 'blue')
ax1.set_xlabel('$x$')
ax1.set_ylabel('$U(x)$')
plt.legend()
#ax1.set_ylim(0, 1)
plt.show()