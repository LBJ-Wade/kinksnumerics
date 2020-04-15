import math
import numpy
import matplotlib.pyplot as plt

x= numpy.linspace(-7,7,1000)
fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
plt.plot(x,numpy.tanh(x),'blue',label='$\phi$')
plt.plot(x,1/numpy.cosh(x)**2,'red',label='$\epsilon$')
ax1.set_xlabel('$x$')
plt.legend()
#ax1.set_ylim(0, 0.01)
plt.show()