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
Nx = 16384  # Number of harmonics
Nt = 10000  # Number of time slices
tmax = 100.0  # Maximum time
c = 0.0#0.1  # Wave speed
a = 3 #two kink seperation
b = 0 #Point between two kinks
dt = tmax / Nt  # time step
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

t = 0.0
u1 = -numpy.tanh(xx-c*t + a)  #<- two kink  #numpy.tanh(xx-c*t) <-single kink
uexact1 = -numpy.tanh(xx-c*t + a) #<- two kink  #numpy.tanh(xx-c*t) <-single kink
uold1 = -numpy.tanh(xx-c*(t-dt) + a)  #<- two kink  #numpy.tanh(xx-c*t) <-single kink
v1 = numpy.fft.fftn(u1)
vold1 = numpy.fft.fftn(uold1)

u2 = numpy.tanh(xx+c*t - a)  #<- two kink  #numpy.tanh(xx-c*t) <-single kink
uexact2 = numpy.tanh(xx+c*t - a) #<- two kink  #numpy.tanh(xx-c*t) <-single kink
uold2 = numpy.tanh(xx+c*(t-dt) - a)  #<- two kink  #numpy.tanh(xx-c*t) <-single kink
v2 = numpy.fft.fftn(u2)
vold2 = numpy.fft.fftn(uold2)

u = u1 + u2 + 1  #<- two kink  #numpy.tanh(xx-c*t) <-single kink
uexact = uexact1 + uexact2 + 1      #<- two kink  #numpy.tanh(xx-c*t) <-single kink
uold = uold1 + uold2 + 1     #<- two kink  #numpy.tanh(xx-c*t) <-single kink
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
InteractionForce = -0.5 * (((u - uold) / dt) ** 2 + (ux) ** 2) + 0.5*((0.5*(u+uold))**4-2*(0.5*(u+uold))**2+1)
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

#EnO = En[0]
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
v0part = []
v0part.append(0)
xt =[3.,
2.999993855762479,
2.999975422747895,
2.999944700050127,
2.9999016861587746,
2.999846378958862,
2.99977877573042,
2.999698873147954,
2.9996066672797865,
2.9995021535872843,
2.999385326923964,
2.999256181534476,
2.9991147110534717,
2.998960908504345,
2.998794766297854,
2.9986162762306194,
2.998425429483503,
2.9982222166198587,
2.9980066275836585,
2.9977786516975,
2.9975382776604773,
2.9972854935459368,
2.997020286799094,
2.996742644234528,
2.9964525520335474,
2.9961499957414164,
2.995834960264459,
2.9955074298670223,
2.995167388168308,
2.994814818139068,
2.9944497020981604,
2.9940720217089676,
2.9936817579756743,
2.9932788912394006,
2.9928634011741955,
2.992435266782878,
2.991994466392739,
2.991540977651086,
2.991074777520639,
2.990595842274777,
2.9901041474926213,
2.989599668053966,
2.9890823781340488,
2.9885522511981546,
2.98800925999606,
2.9874533765563056,
2.986884572180301,
2.9863028174362554,
2.9857080821529354,
2.98510033541324,
2.9844795455475968,
2.983845680127174,
2.9831987059569056,
2.98253858906832,
2.9818652947121818,
2.9811787873509314,
2.9804790306509226,
2.979765987474458,
2.979039619871614,
2.9782998890718493,
2.9775467554754034,
2.9767801786444688,
2.9760001172941366,
2.975206529283118,
2.9743993716042216,
2.973578600374603,
2.9727441708257594,
2.97189603729328,
2.971034153206342,
2.9701584710769433,
2.969268942488873,
2.968365518086409,
2.967448147562737,
2.9665167796480896,
2.9655713620975925,
2.964611841678818,
2.9636381641590304,
2.9626502742921277,
2.961648115805265,
2.960631631385151,
2.9596007626640164,
2.9585554502052425,
2.9574956334886404,
2.956421250895378,
2.95533223969254,
2.9542285360173186,
2.953110074860817,
2.9519767900514684,
2.950828614238049,
2.949665478872282,
2.9484873141910217,
2.947294049198005,
2.9460856116451595,
2.944861928013459,
2.9436229234933156,
2.942368521964488,
2.941098645975509,
2.9398132167226008,
2.938512154028083,
2.9371953763182455,
2.9358628006006797,
2.9345143424410525,
2.9331499159393037,
2.9317694337052598,
2.930372806833638,
2.9289599448784314,
2.927530755826657,
2.9260851460714417,
2.9246230203844408,
2.9231442818875566,
2.921648832023949,
2.9201365705283093,
2.918607395396378,
2.917061202853692,
2.915497887323527,
2.913917341394022,
2.9123194557844556,
2.910704119310652,
2.9090712188494905,
2.907420639302491,
2.905752263558449,
2.9040659724550903,
2.9023616447397163,
2.900639157028809,
2.898898383766564,

2.8971391971823177,

2.8953614672468366,

2.89356506162743,

2.8917498456418507,

2.8899156822109475,

2.8880624318100248,

2.886189952418875,

2.8842980994704353,

2.8823867257980234,

2.880455681581116,

2.878504814289611,

2.8765339686265254,

2.874542986469089,

2.8725317068081604,

2.8704999656859296,

2.868447596131833,

2.866374428096634,

2.8642802883845957,

2.862165000583686,

2.8600283849937482,

2.857870258552562,

2.8556904347597265,

2.853488723598284,

2.851264931454009,

2.8490188610322784,

2.846750311272431,

2.8444590772595393,

2.842144950133481,

2.839807716995233,

2.837447160810268,

2.8350630603089564,

2.832655189883864,

2.8302233194838227,

2.827767214504655,

2.825286635676433,

2.8227813389471277,

2.8202510753625236,

2.8176955909422414,

2.8151146265517295,

2.812507917770059,

2.80987519475336,

2.8072161820937294,

2.8045305986734226,

2.8018181575141474,

2.7990785656212585,

2.7963115238226437,

2.793516726602092,

2.790693861926903,

2.7878426110695123,

2.7849626484228733,

2.782053641309339,

2.7791152497827576,

2.7761471264235094,

2.773148916126164,

2.7701202558794455,

2.7670607745381735,

2.7639700925868196,

2.760847821894313,

2.757693565459704,

2.754506917148269,

2.751287461417636,

2.7480347730334596,

2.744748416774176,

2.7414279471243304,

2.738072907955933,

2.7346828321972967,

2.731257241488746,

2.7277956458245787,

2.7242975431806107,

2.7207624191266158,

2.7171897464228967,

2.71357898460023,

2.7099295795223344,

2.706240962929998,

2.7025125519659263,

2.6987437486793278,

2.6949339395092005,

2.691082494745195,

2.687188767964894,

2.6832520954462513,

2.679271795553864,

2.675247168097677,

2.6711774936626016,

2.6670620329074803,

2.6629000258316724,

2.6586906910074717,

2.654433224776415,

2.6501268004074343,

2.645770567214657,

2.641363649632503,

2.636905146245591,

2.632394128770761,

2.62782964098837,

2.6232106976197764,

2.618536283147747,

2.613805350576261,

2.609016820125939,

2.604169577861046,

2.5992624742437136,

2.594294322610711,

2.5892638975677316,

2.5841699332957617,

2.5790111217637213,

2.573786110841056,

2.568493502303501,

2.5631318497246762,

2.557699656245571,

2.5521953722133524,

2.546617392680176,

2.5409640547519476,

2.535233634776082,

2.5294243453563956,

2.5235343321822197,

2.5175616706576833,

2.511504362315873,

2.5053603310011625,

2.4991274188015176,

2.4928033817108393,

2.4863858849995806,

2.4798724982697626,

2.473260690168208,

2.466547822729269,

2.4597311453154216,

2.452807788120956,

2.4457747552003943,

2.4386289169792863,

2.43136700220059,

2.4239855892547886,

2.416481096836297,

2.408849773862329,

2.401087688583252,

2.393190716805351,

2.3851545291377287,

2.376974577164709,

2.368646078433201,

2.360164000131064,

2.3515230413170607,

2.3427176135453553,

2.333741819707325,

2.3245894308901347,

2.3152538610247575,

2.305728139065089,

2.296004878403907,

2.2860762431896653,

2.2759339111594534,

2.2655690325465905,

2.254972184554595,

2.2441333208108265,

2.233041715120458,

2.2216858987316934,

2.2100535901927207,

2.19813161672513,

2.185905825851978,

2.1733609857940825,

2.1604806728767874,

2.14724714385981,

2.1336411907006907,

2.1196419747691992,

2.1052268369219056,

2.0903710790921557,

2.075047712109992,

2.059227163285835,

2.04287693579945,

2.0259612100364457,

2.00844037457898,

1.990270471408648,

1.9714025357742124,

1.9517818057746297,

1.93134676952902,

1.910028008164279,

1.887746779754905,

1.8644132713401746,

1.83992442104414,

1.814161176831183,

1.786985007436462,

1.7582334064591607,

1.7277140194988532,

1.695196855060751,

1.6604037762399897,

1.6229940479505696,

1.582544017709059,

1.5385178180432728,

1.490223864292105,

1.4367479873596813,

1.3768463097846926,

1.3087647499349795,

1.2299151338138903,

1.1362449234236254,

1.0208697483442009,

0.8706081093043916,

0.6547626600270533,

0.2666247213211003,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,
0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

0.,

]


#
us = [u]
uex = [uexact] #here the exact solution if we apply the force
uerror = [abs((u - uexact)/uexact)*100]
for i in range(len(uerror[0])):
    if abs(uerror[0][i]) > 10**5:
        uerror[0][i] = 10**5

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


    # analytical interaction force
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

    """ax = fig.add_subplot(211)
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
    plt.draw()"""
    #plt.savefig(str(name[ntt]))############## for png saving
    us.extend([u])

    vx = 0.5 * kxm * (v + vold)
    ux = numpy.real(numpy.fft.ifftn(vx))
    Kineticenergy = 0.5 * ((u-uold) / dt) ** 2
    Strainenergy = 0.5 * (ux) ** 2
    Potentialenergy = 0.5*((0.5*(u+uold))**4-2*(0.5*(u+uold))**2+1)#0.5 * (0.5 * (u + uold)) ** 2 - Es * 0.25 * (0.5 * (u + uold)) ** 4
    InteractionForce = -0.5 * (((u - uold) / dt) ** 2 + (ux) ** 2) + 0.5*((0.5*(u+uold))**4-2*(0.5*(u+uold))**2+1)
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
    EnKin[plotnum] = Ekinsum #numpy.real(Kineticenergy[0])
    EnPot[plotnum] = Epotsum #numpy.real(Potentialenergy[0])
    EnStr[plotnum] = Estrsum #numpy.real(Strainenergy[0])
    En[plotnum] = EnStr[plotnum] + EnPot[plotnum] + EnKin[plotnum]
    #Enchange[plotnum - 1] = numpy.log(abs(1 - En[plotnum] / EnO))
    deltat= t-tdata[plotnum-1]
    v0 = 0
    for i in range(len(v0part)):
        v0 = v0 + v0part[i]
    #a = 0.5*(2*a - 2 * v0*deltat - 2 * 0.5 * Fana[plotnum] * deltat**2)
    gamma=1/numpy.sqrt(1-v0**2)
    #uexact = -numpy.tanh(gamma*(xx + a)) + numpy.tanh(gamma*(xx - a)) + 1  # <- two kink  #numpy.tanh(xx-c*t) <-single kink

    tdata[plotnum] = t
    v0part.append( Fana[plotnum] * deltat)

    #xt = 0.5*numpy.log(     (numpy.sinh(8*numpy.imag*numpy.sqrt(*numpy.exp(4*a))*(1/16*numpy.exp(2*a)*numpy.pi+t)))/(numpy.sqrt(numpy.exp(-4*a))*numpy.imag)   )
    #print(xt)
    uexact = -numpy.tanh(gamma * (xx + xt[plotnum])) + numpy.tanh(gamma * (xx - xt[plotnum])) + 1

    uex.extend([uexact])
    uerror.extend([abs((u - uexact)/uexact)*100])
    ntt += 1
    for i in range(len(uerror[plotnum])):
        if abs(uerror[plotnum][i]) > 10**5:
            uerror[plotnum][i] = 10**5

plt.ioff()
plt.show()


#animation on the interaction energy between two kink

"""fig = plt.figure()
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
ax1.set_ylim(-2, 3)
ax1.set_xlim(-10, 10)
ax2.set_ylim(0, 1)
ax2.set_xlim(-10, 10)
plt.xlabel('x')
plt.ylabel('$|\\frac{\phi_{simulated}-\phi_{exact}}{\phi_{exact}}|\,\,\,\,\,[10^{-2}]$')
#plt.hlines(0.1,-10,10,colors='black',linestyles='--')


def update(i):
    new_data1 = us[i:i+1]
    line1.set_ydata(new_data1)
    new_data1a = uex[i:i+1]
    line1a.set_ydata(new_data1a)
    new_data2 = uerror[i:i+1]
    line2.set_ydata(new_data2)
    return line1, line2,


ani = animation.FuncAnimation(fig, update, frames=numplots, interval=55)
plt.legend()
plt.show()
ani.save('two_kink_attraction_corrected_simulated_force.gif', writer='D_M')

















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