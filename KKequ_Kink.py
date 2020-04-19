import math
import numpy
import matplotlib.pyplot as plt
from matplotlib import animation
import scipy.integrate as integrate
import time
def truncate(n, decimals=0):
    multiplier = 10 ** decimals
    return int(n * multiplier) / multiplier



plt.ion()

# Grid
Lx = 128.0  # Period 2*pi*Lx
Nx = 8192  # Number of harmonics
Nt = 10000  # Number of time slices
tmax = 100.0  # Maximum time
c = 0.1#0.1  # Wave speed
gamma = 1/numpy.sqrt(1-c**2)
a = 3 #two kink seperation
dt = tmax / Nt  # time step
plotgap = 50  # time steps between plots
numplots = Nt / plotgap  # number of plots to make

x = [i * 2.0 * math.pi * (Lx / Nx) for i in xrange(-Nx / 2, 1 + Nx / 2)]
k_x = (1.0 / Lx) * numpy.array([complex(0, 1) * n for n in range(0, Nx / 2) \
                                + [0] + range(-Nx / 2 + 1, 0)])

kxm = numpy.zeros((Nx), dtype=complex)
xx = numpy.zeros((Nx), dtype=float)

for i in xrange(Nx):
    kxm[i] = k_x[i]
    xx[i] = x[i]

dx = abs(xx[4096]-xx[4097])
# allocate arrays
unew = numpy.zeros((Nx), dtype=float)
u = numpy.zeros((Nx), dtype=float)
ut = numpy.zeros((Nx), dtype=float)
uexact = numpy.zeros((Nx), dtype=float)
uold = numpy.zeros((Nx), dtype=float)
vnew = numpy.zeros((Nx), dtype=complex)
v = numpy.zeros((Nx), dtype=complex)
vold = numpy.zeros((Nx), dtype=complex)
ux = numpy.zeros((Nx), dtype=float)
vx = numpy.zeros((Nx), dtype=complex)
InteractionForce = numpy.zeros((Nx), dtype=complex)
Kineticenergy = numpy.zeros((Nx), dtype=complex)
Potentialenergy = numpy.zeros((Nx), dtype=complex)
Strainenergy = numpy.zeros((Nx), dtype=complex)
Kineticenergyana = numpy.zeros((Nx), dtype=complex)
Potentialenergyana = numpy.zeros((Nx), dtype=complex)
Strainenergyana = numpy.zeros((Nx), dtype=complex)
EnKin = numpy.zeros((numplots), dtype=float)
EnPot = numpy.zeros((numplots), dtype=float)
EnStr = numpy.zeros((numplots), dtype=float)
En = numpy.zeros((numplots), dtype=float)
Enchange = numpy.zeros((numplots - 1), dtype=float)
tdata = numpy.zeros((numplots), dtype=float)
nonlin = numpy.zeros((Nx), dtype=float)
nonlinhat = numpy.zeros((Nx), dtype=complex)

t = 0.0
u = numpy.tanh(gamma*(xx-c*t)) #<-single kink
uexact = numpy.tanh(gamma*(xx-c*t)) #<-single kink
uold = numpy.tanh(gamma*(xx-c*(t-dt))) #<-single kink
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
#plt.show()

# initial energy
vx = 0.5 * kxm * (v + vold)
ux = numpy.real(numpy.fft.ifftn(vx))
Kineticenergy = 0.5 * ((u - uold) / dt) ** 2
Strainenergy = 0.5* (ux) ** 2
Potentialenergy = 0.5*((0.5*(u+uold))**4-2*(0.5*(u+uold))**2+1)#0.5 * (0.5 * (u + uold)) ** 2 - Es * 0.25 * (0.5 * (u + uold)) ** 4
#InteractionForce =
fKineticenergy = [Kineticenergy]
fPotentialenergy = [Potentialenergy]
fStrainenergy = [Strainenergy]
#sum over an reasonable intervall
Ekinsum = 0
Epotsum = 0
Estrsum = 0
for i in range(len(xx)):
    if (xx[i] > -90) & (xx[i] < 90):
        Ekinsum = Ekinsum + Kineticenergy[i]*dx
        Epotsum = Epotsum + Potentialenergy[i]*dx
        Estrsum = Estrsum + Strainenergy[i]*dx

EnKin[0] = Ekinsum
EnPot[0] = Epotsum
EnStr[0] = Estrsum
En[0] = EnStr[0] + EnPot[0] + EnKin[0]

#EnO = En[0]
tdata[0] = t
plotnum = 0


#analytical                                           ######################old########################################################
Kineticenergyana = 0.5 * (-c * 1 / (numpy.cosh(c * t - xx)) ** 2) ** 2
Strainenergyana = 0.5 * (1/(numpy.cosh(-c * t + xx))**2)**2
Potentialenergyana = 0.5 * ((numpy.tanh(xx - c * t)) ** 4 - 2 * (numpy.tanh(xx - c * t)) ** 2 + 1)
fKineticenergyana = [Kineticenergyana]
fPotentialenergyana = [Potentialenergyana]
fStrainenergyana = [Strainenergyana]
fPotentialenergyerror = [abs(Strainenergyana-Strainenergy)]
#integrate over everything for analyticall
EnKinana = [integrate.quad(lambda z: 0.5 * (-c * 1 / (numpy.cosh(c * t - z)) ** 2) ** 2, -numpy.inf, numpy.inf)[0]]
EnStrana = [integrate.quad(lambda z: 0.5 * ((numpy.cosh(-c * t + z)) ** -2) ** 2, -numpy.inf, numpy.inf)[0]]
EnPotana = [integrate.quad(lambda z: 0.5 * ((numpy.tanh(z - c * t)) ** 4 - 2 * (numpy.tanh(z - c * t)) ** 2 + 1), -numpy.inf, numpy.inf)[0]]
Enana = [EnStrana[0] + EnPotana[0] + EnKinana[0]]

"""fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2)
ax1.plot(xx, Potentialenergy, 'b-', label='$\phi$')
ax1.plot(xx, 0.5 * ((numpy.tanh(xx - c * t)) ** 4 - 2 * (numpy.tanh(xx - c * t)) ** 2 + 1), 'r-', label='$\phi_e$')
ax1.set_xlim(-10, 10)
ax2.plot(xx, Kineticenergy, 'b-', label='$\phi$')
ax2.plot(xx, 0.5 * (-c * 1 / (numpy.cosh(c * t - xx)) ** 2) ** 2, 'r-', label='$\phi_e$')
ax2.set_xlim(-10, 10)
plt.ioff()
plt.show()"""

# plot each plot in a png
#name = ["0.png"]
#for i in range(numplots):
#    name.extend([str(i+1)+".png"])
#    print(name[i])
ntt = 0
#
us = [u]
uex = [uexact]
uerror = [abs((u - uexact)/uexact)*1000]
for i in range(len(uerror[0])):
    if abs(uerror[0][i]) > 10**5:
        uerror[0][i] = uexact[i]

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
    uexact = numpy.tanh(gamma*(xx-c*t)) #<-single kink
    ax = fig.add_subplot(211)
    plt.cla()
    ax.plot(xx, u, 'b-', label='$\phi$')
    ax.plot(xx, uexact, 'r-', label='$\phi_{exact}$')
    #plt.xlim(-10, 10)
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
    uex.extend([uexact])
    uerror.extend([abs((u - uexact)/uexact)*1000])
    ntt += 1
    for i in range(len(uerror[plotnum])):
        if abs(uerror[plotnum][i]) > 10**5:
            uerror[plotnum][i] = uexact[i]

    vx = 0.5 * kxm * (v + vold)
    ux = numpy.real(numpy.fft.ifftn(vx))
    Kineticenergy = 0.5 * ((u-uold) / dt) ** 2
    Strainenergy = 0.5 * (ux) ** 2
    Potentialenergy = 0.5*((0.5*(u+uold))**4-2*(0.5*(u+uold))**2+1)#0.5 * (0.5 * (u + uold)) ** 2 - Es * 0.25 * (0.5 * (u + uold)) ** 4
    fKineticenergy.extend([Kineticenergy])
    fPotentialenergy.extend([Potentialenergy])
    fStrainenergy.extend([Strainenergy])
    #analytical                                       ######################old########################################################
    Kineticenergyana = 0.5 * (-c * 1 / (numpy.cosh(c * t - xx)) ** 2) ** 2
    Strainenergyana = 0.5 * (1/(numpy.cosh(-c * t + xx))**2)**2
    Potentialenergyana = 0.5 * ((numpy.tanh(xx - c * t)) ** 4 - 2 * (numpy.tanh(xx - c * t)) ** 2 + 1)
    fKineticenergyana.extend([Kineticenergyana])
    fPotentialenergyana.extend([Potentialenergyana])
    fStrainenergyana.extend([Strainenergyana])
    fPotentialenergyerror.extend([abs(Strainenergyana - Strainenergy)])

    EnKinana.extend([integrate.quad(lambda x: 0.5 * (-c * 1/(numpy.cosh(c * t - x))**2)**2, -numpy.inf, numpy.inf)[0]])
    EnStrana.extend([integrate.quad(lambda x: 0.5 * (1/(numpy.cosh(-c * t + x))**2)**2, -numpy.inf, numpy.inf)[0]])
    EnPotana.extend([integrate.quad(lambda x: 0.5*((numpy.tanh(x-c*t))**4-2*(numpy.tanh(x-c*t))**2+1), -numpy.inf, numpy.inf)[0]])
    Enana.extend([integrate.quad(lambda x: 0.5 * (-c * 1/(numpy.cosh(c * t - x))**2)**2 + 0.5 * (1/(numpy.cosh(-c * t + x))**2)**2 + 0.5*((numpy.tanh(x-c*t))**4-2*(numpy.tanh(x-c*t))**2+1), -numpy.inf, numpy.inf)[0]])

    """ax = fig.add_subplot(212)
    plt.cla()
    ax.plot(xx, Kineticenergy, 'b-')  # ax.plot(xx, uexact, 'b-')
    ax.plot(xx, Kineticenergyana, 'r-')
    plt.xlim(-10, 10)
    plt.ylim(0, 0.02)
    plt.xlabel('x')
    plt.ylabel('Epot')
    plt.draw()"""

    #Kineticenergy = numpy.fft.fftn(Kineticenergy)
    #Strainenergy = numpy.fft.fftn(Strainenergy)
    #Potentialenergy = numpy.fft.fftn(Potentialenergy)
    Ekinsum = 0
    Epotsum = 0
    Estrsum = 0
    for i in range(len(xx)):
        if (xx[i] > -15) & (xx[i] < 15):
            Ekinsum = Ekinsum + Kineticenergy[i]*dx
            Epotsum = Epotsum + Potentialenergy[i]*dx
            Estrsum = Estrsum + Strainenergy[i]*dx
    EnKin[plotnum] = Ekinsum #numpy.real(Kineticenergy[0])
    EnPot[plotnum] = Epotsum #numpy.real(Potentialenergy[0])
    EnStr[plotnum] = Estrsum #numpy.real(Strainenergy[0])
    En[plotnum] = EnStr[plotnum] + EnPot[plotnum] + EnKin[plotnum]
    #Enchange[plotnum - 1] = numpy.log(abs(1 - En[plotnum] / EnO))
    tdata[plotnum] = t

plt.ioff()
plt.show()




#animation of live quantities, energy and energy split
"""
fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2)

ax1.set_ylabel('$E_{Total}$')
#ax1.set_xlim(-15, 15)
ax1.set_ylim(1.339, 1.341)
ax2.set_xlabel('t')
ax2.set_ylabel('$E_{Kin}$')
#ax2.set_xlim(-15, 15)
ax2.set_ylim(0.006, 0.008)
lines = []
for i in range(len(tdata)):
    line1,  = ax1.plot(tdata[:i], En[:i], color='blue')
    line1a,  = ax1.plot(tdata[:i], Enana[:i], color='red')
    line2,  = ax2.plot(tdata[:i], EnKin[:i], color='blue', label='$E_{Kin,sim}$')
    line2a, = ax2.plot(tdata[:i], EnKinana[:i], color='black', label='$E_{Kin,ana}$')
    #line2b,  = ax2.plot(tdata[:i], EnPot[:i], color='red', label='$E_{Pot,sim}$')
    #line2c, = ax2.plot(tdata[:i], EnPotana[:i], color='black', label='$E_{Pot,ana}$')
    #line2d,  = ax2.plot(tdata[:i], EnStr[:i], color='blue', label='$E_{Str,sim}$')
    #line2e, = ax2.plot(tdata[:i], EnStrana[:i], color='green', label='$E_{Str,ana}$')
    lines.append([line1, line1a, line2, line2a])


#line1, = ax1.plot(xx, fStrainenergy[0], 'r-')
#line1a, = ax1.plot(xx, fStrainenergyana[0], 'b-')
#line2, = ax2.plot(xx, fPotentialenergyerror[0], 'b-')
#def update(i):
#    new_data1 = fStrainenergy[i:i+1]
#    line1.set_ydata(new_data1)
#    new_data1a = fStrainenergyana[i:i+1]
#    line1a.set_ydata(new_data1a)
#    new_data2 = fPotentialenergyerror[i:i+1]
#    line2.set_ydata(new_data2)
#    return line1a, line1, line2,
#
#ani = animation.FuncAnimation(fig, update, frames=numplots, interval=75)
#plt.show()
ani = animation.ArtistAnimation(fig, lines, interval=50, blit=True)
plt.show()

ani.save('totalenergy_and_split_kinetic.gif', writer='D_M')
"""





#animation of the simulated solution and error

fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
plt.ylabel('$\phi_{simulated}$')
ax2 = fig.add_subplot(2, 1, 2)
line1, = ax1.plot(xx, us[0], 'r-', label='$\phi$')
line1a, = ax1.plot(xx, uex[0], 'b-', label='$\phi_{exact}$')
line2, = ax2.plot(xx, uerror[0], 'b-')
plt.legend()
ax1.set_ylim(-2, 3)
ax1.set_xlim(-10, 10)
ax2.set_ylim(0, 1)
ax2.set_xlim(-10, 10)
plt.xlabel('x')
plt.ylabel('$|\\frac{\phi_{simulated}-\phi_{exact}}{\phi_{exact}}|\,\,\,\,\,[10^{-3}]$')
#plt.hlines(0.001,-10,10,colors='black',linestyles='--')


def update(i):
    new_data1 = us[i:i+1]
    line1.set_ydata(new_data1)
    new_data1a = uex[i:i+1]
    line1a.set_ydata(new_data1a)
    new_data2 = uerror[i:i+1]
    line2.set_ydata(new_data2)
    return line1, line2,


ani = animation.FuncAnimation(fig, update, frames=numplots, interval=75)
plt.legend()
plt.show()
ani.save('one_kink_moving.gif', writer='D_M')

















# animation of the exact solution #old
"""
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
"""


#old stuff
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