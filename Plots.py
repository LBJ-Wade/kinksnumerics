import math
import numpy
import matplotlib.pyplot as plt
x= numpy.linspace(-14,14,1000)
fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
plt.plot(x, (1-numpy.cos(x)), 'blue')
ax1.set_xlabel('$\phi$')
ax1.set_ylabel('$U(\phi)$')
plt.legend()
ax1.set_ylim(0, 2.5)
plt.show()