import math
import numpy
import matplotlib.pyplot as plt
from matplotlib import animation
import scipy.integrate as integrate


plt.ion()

# Grid
Lx = 128.0  # Period 2*pi*Lx
Nx = 16384  # Number of harmonics
Nt = 10000  # Number of time slices
tmax = 50.0  # Maximum time
c = 0.1# 0.1  # Wave speed
gamma=1/numpy.sqrt(1-c**2)
a = 3 # two kink seperation
b = 0 # Point between two kinks
dt = (tmax+50) / Nt  # time step
plotgap = 25  # time steps between plots
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
print(dx)

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
unew1 = numpy.zeros((Nx), dtype=float)
u1 = numpy.zeros((Nx), dtype=float)
ut1 = numpy.zeros((Nx), dtype=float)
uexact1 = numpy.zeros((Nx), dtype=float)
uold1 = numpy.zeros((Nx), dtype=float)
vnew1 = numpy.zeros((Nx), dtype=complex)
v1 = numpy.zeros((Nx), dtype=complex)
vold1 = numpy.zeros((Nx), dtype=complex)
ux1 = numpy.zeros((Nx), dtype=float)
vx1 = numpy.zeros((Nx), dtype=complex)
unew2 = numpy.zeros((Nx), dtype=float)
u2 = numpy.zeros((Nx), dtype=float)
ut2 = numpy.zeros((Nx), dtype=float)
uexact2 = numpy.zeros((Nx), dtype=float)
uold2 = numpy.zeros((Nx), dtype=float)
vnew2 = numpy.zeros((Nx), dtype=complex)
v2 = numpy.zeros((Nx), dtype=complex)
vold2 = numpy.zeros((Nx), dtype=complex)
ux2 = numpy.zeros((Nx), dtype=float)
vx2 = numpy.zeros((Nx), dtype=complex)
InteractionForce = numpy.zeros((Nx), dtype=complex)
F = numpy.zeros((numplots), dtype=float)
Fana = numpy.zeros((numplots), dtype=float)
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

t = -50.0 # t=0 is the piont of collision

u = 4*numpy.arctan(-c*numpy.cosh(gamma*xx)/numpy.sinh(gamma*c*t)) #<- two kink
uexact = 4*numpy.arctan(-c*numpy.cosh(gamma*xx)/numpy.sinh(gamma*c*t))  #<- two kink
uold = 4*numpy.arctan(-c*numpy.cosh(gamma*xx)/numpy.sinh(gamma*c*(t-dt)))     #<- two kink
v = numpy.fft.fftn(u)
vold = numpy.fft.fftn(uold)

#Not shown 
fig = plt.figure()
ax = fig.add_subplot(211)
ax.plot(xx, u, 'b-')
plt.xlim(-20, 20)
plt.ylim(-7, 7)
plt.xlabel('x')
plt.ylabel('u')
ax = fig.add_subplot(212)
ax.plot(xx, abs(u - uexact), 'b-')
plt.xlim(-20, 20)
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
Potentialenergy = 1-numpy.cos(u)
fKineticenergy = [Kineticenergy]
fPotentialenergy = [Potentialenergy]
fStrainenergy = [Strainenergy]

#sum over an reasonable intervall
Ekinsum = 0
Epotsum = 0
Estrsum = 0
IntFsum = 0
for i in range(len(xx)):
    if (xx[i] > -90) & (xx[i] < 90):
        Ekinsum = Ekinsum + Kineticenergy[i]*dx
        Epotsum = Epotsum + Potentialenergy[i]*dx
        Estrsum = Estrsum + Strainenergy[i]*dx

for i in range(len(xx)):
    if (xx[i] > b-dx/2) & (xx[i] <= b+dx/2):
        IntFsum = InteractionForce[i]

F[0] = IntFsum
EnKin[0] = Ekinsum
EnPot[0] = Epotsum
EnStr[0] = Estrsum
En[0] = EnStr[0] + EnPot[0] + EnKin[0]

tdata[0] = t
plotnum = 0


#analytical interaction force
x1 = 0
x2 = 0
counter1 = 0
counter2 = 0
for i in range(len(xx)):
    if (counter1 < 2) & (xx[i] < 0) & (u[i] < 0.0+0.05) & (u[i] > 0.0-0.05):
        x1 = xx[i]
        counter1 = counter1 + 1

    if (counter2 < 2) & (xx[i] > 0) & (u[i] < 0.0 + 0.05) & (u[i] > 0.0 - 0.05):
        x2 = xx[i]
        counter2 = counter2 + 1
R = abs(x1-x2)
Fana[0] = 32 * numpy.exp(-2*2*a)

#analytical energy
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


v0part = []
v0part.append(0)


us = [u]
uex = [uexact]
uerror = [abs((u - uexact)/uexact)*1000]
for i in range(len(uerror[0])):
    if abs(uexact[i]) < 10**-7:
        uerror[0][i] = uexact[i]

for nt in xrange(numplots - 1):
    for n in xrange(plotgap):
        nonlin = numpy.sin(u)
        nonlinhat = numpy.fft.fftn(nonlin)
        vnew = (-nonlinhat+1/(dt*dt)*(2*v-vold)+0.25*kxm**2*(2*v+vold))/(1/(dt*dt)-kxm**2*0.25)#((0.25 * (kxm ** 2 - 1) * (2 * v + vold) + (2 * v - vold) / (dt * dt) + Es * nonlinhat) /(1 / (dt * dt) - (kxm ** 2 - 1) * 0.25))
        unew = numpy.real(numpy.fft.ifftn(vnew))
        t += dt
        # update old terms
        vold = v
        v = vnew
        uold = u
        u = unew
    plotnum += 1

    deltat = t - tdata[plotnum - 1]
    uexact = 4 * numpy.arctan(-c * numpy.cosh(gamma * xx) / numpy.sinh(gamma * c * t))
    # analytical interaction force (discretized)
    x1 = 0
    x2 = 0
    uj1 = 1
    uj2 = 1
    for i in range(len(xx)):
        if (xx[i] < 0) & (u[i] > -0.05) & (uj1 > u[i]):
            x1 = xx[i]
            uj1 = u[i]
        if (xx[i] > 0) & (u[i] > -0.05) & (uj2 > u[i]):
            x2 = xx[i]
            uj2 = u[i]
    R = abs(x1 - x2)

    Fana[plotnum] = 32*numpy.exp(-2*R)

    us.extend([u])
    uex.extend([uexact])

    uerror.extend([abs((u - uexact)/uexact)*1000])
    for i in range(len(uerror[plotnum])):
        if abs(uerror[plotnum][i]) > 10**5:
            uerror[plotnum][i] = uexact[i]

    vx = 0.5 * kxm * (v + vold)
    ux = numpy.real(numpy.fft.ifftn(vx))
    Kineticenergy = 0.5 * ((u-uold) / dt) ** 2
    Strainenergy = 0.5 * (ux) ** 2
    Potentialenergy = 0.5*((0.5*(u+uold))**4-2*(0.5*(u+uold))**2+1)
    InteractionForce = -0.5 * (((u - uold) / dt) ** 2 + (ux) ** 2) + 0.5*((0.5*(u+uold))**4-2*(0.5*(u+uold))**2+1)
    fKineticenergy.extend([Kineticenergy])
    fPotentialenergy.extend([Potentialenergy])
    fStrainenergy.extend([Strainenergy])
    #analytical
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

    Ekinsum = 0
    Epotsum = 0
    Estrsum = 0
    IntFsum = 0

    for i in range(len(xx)):
        if (xx[i] > -15) & (xx[i] < 15):
            Ekinsum = Ekinsum + Kineticenergy[i]*dx
            Epotsum = Epotsum + Potentialenergy[i]*dx
            Estrsum = Estrsum + Strainenergy[i]*dx

    for i in range(len(xx)):
        if (xx[i] > b - dx / 2) & (xx[i] <= b + dx / 2):
            IntFsum = InteractionForce[i]

    F[plotnum] = IntFsum
    EnKin[plotnum] = Ekinsum
    EnPot[plotnum] = Epotsum
    EnStr[plotnum] = Estrsum
    En[plotnum] = EnStr[plotnum] + EnPot[plotnum] + EnKin[plotnum]

    tdata[plotnum] = t
    v0part.append( Fana[plotnum] * deltat)

plt.ioff()
plt.show()


#animation on the interaction energy between two kink
"""
fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
#ax2 = fig.add_subplot(2, 1, 2)
ax1.set_ylabel('$F_{interaction}(t=0...85)$')
ax1.set_ylim(0, 0.01)
lines = []
for i in range(len(tdata)):
    line1,  = ax1.plot(tdata[:i], F[:i], color='red')
    line1a,  = ax1.plot(tdata[:i], Fana[:i], color='blue')
    lines.append([line1, line1a])

ani = animation.ArtistAnimation(fig, lines, interval=45, blit=True)
plt.show()"""
#ani.save('interaction_force_two_kink.gif', writer='D_M')

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
line1a, = ax1.plot(xx, uex[0], 'b-', label='$\phi_{exact}$' )
line2, = ax2.plot(xx, uerror[0], 'b-')
plt.legend()
ax1.set_ylim(-7, 14)
ax1.set_xlim(-20, 20)
ax2.set_ylim(0, 1)
ax2.set_xlim(-20, 20)
plt.xlabel('x')
plt.ylabel('$|\\frac{\phi_{simulated}-\phi_{exact}}{\phi_{exact}}|\,\,\,\,\,[10^{-2}]$')



def update(i):
    new_data1 = us[i:i+1]
    line1.set_ydata(new_data1)

    new_data1a = uex[i:i+1]
    line1a.set_ydata(new_data1a)
    new_data2 = uerror[i:i+1]
    line2.set_ydata(new_data2)
    return line1, line2,


ani = animation.FuncAnimation(fig, update, frames=numplots, interval=35)
plt.legend()
plt.show()
#ani.save('two_kink_sine_gordon.gif', writer='D_M')


