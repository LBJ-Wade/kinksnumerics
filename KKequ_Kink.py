import math
import numpy
import matplotlib.pyplot as plt
from matplotlib import animation
import time

plt.ion()

# Grid
Lx = 64.0  # Period 2*pi*Lx
Nx = 4096  # Number of harmonics
Nt = 3000  # Number of time slices
tmax = 50.0  # Maximum time
c = 0.2  # Wave speed
dt = tmax / Nt  # time step
plotgap = 10  # time steps between plots
Es = 1.0  # focusing (+1) or defocusing (-1) parameter
numplots = Nt / plotgap  # number of plots to make

x = [i * 2.0 * math.pi * (Lx / Nx) for i in xrange(-Nx / 2, 1 + Nx / 2)]
k_x = (1.0 / Lx) * numpy.array([complex(0, 1) * n for n in range(0, Nx / 2) \
                                + [0] + range(-Nx / 2 + 1, 0)])

kxm = numpy.zeros((Nx), dtype=complex)
xx = numpy.zeros((Nx), dtype=float)

for i in xrange(Nx):
    kxm[i] = k_x[i]
    xx[i] = x[i]

# allocate arrays
unew = numpy.zeros((Nx), dtype=float)
u = numpy.zeros((Nx), dtype=float)
uexact = numpy.zeros((Nx), dtype=float)
uold = numpy.zeros((Nx), dtype=float)
vnew = numpy.zeros((Nx), dtype=complex)
v = numpy.zeros((Nx), dtype=complex)
vold = numpy.zeros((Nx), dtype=complex)
ux = numpy.zeros((Nx), dtype=float)
vx = numpy.zeros((Nx), dtype=complex)
Kineticenergy = numpy.zeros((Nx), dtype=complex)
Potentialenergy = numpy.zeros((Nx), dtype=complex)
Strainenergy = numpy.zeros((Nx), dtype=complex)
EnKin = numpy.zeros((numplots), dtype=float)
EnPot = numpy.zeros((numplots), dtype=float)
EnStr = numpy.zeros((numplots), dtype=float)
En = numpy.zeros((numplots), dtype=float)
Enchange = numpy.zeros((numplots - 1), dtype=float)
tdata = numpy.zeros((numplots), dtype=float)
nonlin = numpy.zeros((Nx), dtype=float)
nonlinhat = numpy.zeros((Nx), dtype=complex)

t = 0.0
u = -numpy.tanh(xx-c*t+5) + numpy.tanh(xx+c*t-5) + 1  #numpy.tanh(xx-c*t) #numpy.sqrt(2) / (numpy.cosh((xx - c * t) / numpy.sqrt(1.0 - c ** 2)))
uexact = -numpy.tanh(xx-c*t+5) + numpy.tanh(xx+c*t-5) + 1  #numpy.tanh(xx-c*t)#numpy.sqrt(2) / (numpy.cosh((xx - c * t) / numpy.sqrt(1.0 - c ** 2)))
uold = -numpy.tanh(xx-c*(t-dt)+5) + numpy.tanh(xx+c*(t-dt)-5) + 1  #numpy.tanh(xx-c*(t-dt))#numpy.sqrt(2) / (numpy.cosh((xx + c * dt) / numpy.sqrt(1.0 - c ** 2)))
v = numpy.fft.fftn(u)
vold = numpy.fft.fftn(uold)


#Not shown 
fig = plt.figure()
ax = fig.add_subplot(211)
ax.plot(xx, u, 'b-')
plt.xlim(-10, 10)
plt.ylim(-1.5, 3.0)
plt.xlabel('x')
plt.ylabel('u')
ax = fig.add_subplot(212)
ax.plot(xx, abs(u - uexact), 'b-')
plt.xlim(-10, 10)
plt.ylim(-1.5, 3.0)
plt.xlabel('x')
plt.ylabel('error')
#plt.ioff()
plt.show()

#"""
# initial energy
vx = 0.5 * kxm * (v + vold)
ux = numpy.real(numpy.fft.ifftn(vx))
Kineticenergy = 0.5 * ((u - uold) / dt) ** 2
Strainenergy = 0.5 * (ux) ** 2
Potentialenergy = 0.5*((0.5*(u+uold))**4-2*(0.5*(u+uold))**2+1)#0.5 * (0.5 * (u + uold)) ** 2 - Es * 0.25 * (0.5 * (u + uold)) ** 4
Kineticenergy = numpy.fft.fftn(Kineticenergy)
Strainenergy = numpy.fft.fftn(Strainenergy)
Potentialenergy = numpy.fft.fftn(Potentialenergy)
EnKin[0] = numpy.real(Kineticenergy[0])
EnPot[0] = numpy.real(Potentialenergy[0])
EnStr[0] = numpy.real(Strainenergy[0])
En[0] = EnStr[0] + EnPot[0] + EnKin[0]
EnO = En[0]
tdata[0] = t
plotnum = 0
# plot each plot in a png
#name = ["0.png"]
#for i in range(numplots):
#    name.extend([str(i+1)+".png"])
#    print(name[i])
ntt = 0
#
us = [u]
for nt in xrange(numplots - 1):
    for n in xrange(plotgap):
        nonlin = u ** 3
        nonlinhat = numpy.fft.fftn(nonlin)
        vnew = (0.5*(2*v + vold) + kxm**2 *0.25* (2*v + vold) - 2*nonlinhat + (2*v-vold)/(dt*dt))/(1/(dt*dt)-0.25*(kxm**2+2))#((0.25 * (kxm ** 2 - 1) * (2 * v + vold) + (2 * v - vold) / (dt * dt) + Es * nonlinhat) /(1 / (dt * dt) - (kxm ** 2 - 1) * 0.25))
        unew = numpy.real(numpy.fft.ifftn(vnew))
        t += dt
        # update old terms
        vold = v
        v = vnew
        uold = u
        u = unew
    plotnum += 1
    uexact = -numpy.tanh(xx-c*t+5) + numpy.tanh(xx+c*t-5) +1 #numpy.sqrt(2) / (numpy.cosh((xx - c * t) / numpy.sqrt(1.0 - c ** 2)))
    ax = fig.add_subplot(211)
    plt.cla()
    ax.plot(xx, u, 'b-', label='$\phi$')
    ax.plot(xx, uexact, 'r-',label='$\phi_{exact}$')
    plt.xlim(-10, 10)
    plt.ylim(-1.5, 3.0)
    plt.title('time t=' + str(t))
    plt.xlabel('x')
    plt.ylabel('$\phi$, $\phi_{exact}$')
    plt.legend()
    ax = fig.add_subplot(212)
    plt.cla()
    ax.plot(xx, abs(u - uexact), 'b-')#ax.plot(xx, uexact, 'b-')
    plt.xlim(-10, 10)
    plt.ylim(-0.5, 0.5)
    plt.xlabel('x')
    plt.ylabel('error')
    plt.draw()
    #plt.savefig(str(name[ntt]))############## for png saving
    us.extend([u])
    ntt += 1

    vx = 0.5 * kxm * (v + vold)
    ux = numpy.real(numpy.fft.ifftn(vx))
    Kineticenergy = 0.5 * ((u - uold) / dt) ** 2
    Strainenergy = 0.5 * (ux) ** 2
    Potentialenergy = 0.5*((0.5*(u+uold))**4-2*(0.5*(u+uold))**2+1)#0.5 * (0.5 * (u + uold)) ** 2 - Es * 0.25 * (0.5 * (u + uold)) ** 4
    Kineticenergy = numpy.fft.fftn(Kineticenergy)
    Strainenergy = numpy.fft.fftn(Strainenergy)
    Potentialenergy = numpy.fft.fftn(Potentialenergy)
    EnKin[plotnum] = numpy.real(Kineticenergy[0])
    EnPot[plotnum] = numpy.real(Potentialenergy[0])
    EnStr[plotnum] = numpy.real(Strainenergy[0])
    En[plotnum] = EnStr[plotnum] + EnPot[plotnum] + EnKin[plotnum]
    Enchange[plotnum - 1] = numpy.log(abs(1 - En[plotnum] / EnO))
    tdata[plotnum] = t

plt.ioff()
plt.show()

# animation of the exact solution
fig = plt.figure()
ax = plt.axes(xlim=(-10, 10), ylim=(-2, 3))
line, = ax.plot([], [], lw=2)

def init():
    line.set_data([], [])
    return line,

def animate(i):
    x = numpy.linspace(-10, 10, 1000)
    y = -numpy.tanh(x-c*i+5) + numpy.tanh(x+c*i-5) + 1
    line.set_data(x, y)
    return line,

anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=200, interval=100, blit=True)

plt.show()


#animation of the simulated solution
"""
tme = 0

fig, ax = plt.subplots()
line, = ax.plot(xx, us[0], 'r-')
ax.set_ylim(-2, 3)
ax.set_xlim(-10, 10)


def update(i):
    new_data = us[i:i+1]
    line.set_ydata(new_data)
    return line,


ani = animation.FuncAnimation(fig, update, frames=numplots, interval=100)
plt.show()
ani.save('Kink_annihilation.gif', writer='D_M')
"""



"""
plt.figure()
plt.plot(tdata, En, 'r+', tdata, EnKin, 'b:', tdata, EnPot, 'g-.', tdata, EnStr, 'y--')
plt.xlabel('Time')
plt.ylabel('Energy')
plt.legend(('Total', 'Kinetic', 'Potential', 'Strain'))
plt.title('Time Dependence of Energy Components')
plt.show()

plt.figure()
plt.plot(Enchange, 'r-')
plt.title('Time Dependence of Change in Total Energy')
plt.show()
"""