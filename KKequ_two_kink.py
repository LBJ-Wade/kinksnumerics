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
tmax = 100.0  # Maximum time
c = 0.0# 0.1  # Wave speed
a = 3 # two kink separation
b = 0 # Point between two kinks
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
Potentialenergy = 0.5*((0.5*(u+uold))**4-2*(0.5*(u+uold))**2+1)
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

#analytical
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


# the following two fields v0part and xt are needed to a) calculate the Lorentz factor in the analytical solution, b) give values of the analytic solution of
# the distance between kink an antikink for each time step
v0part = []
v0part.append(0)
xt =[3.,

2.999995391826578,

2.9999815671364276,

2.999958525419867,

2.999926265827346,

2.999884787169316,

2.999834087916059,

2.9997741661974584,

2.9997050198027257,

2.9996266461800714,

2.9995390424363304,

2.9994422053365333,

2.999336131303429,

2.9992208164169543,

2.999096256413655,

2.9989624466860536,

2.998819382281966,

2.99866705790377,

2.9985054679076133,

2.998334606302581,

2.9981544667497992,

2.9979650425614963,

2.9977663267000025,

2.9975583117767015,

2.9973409900509274,

2.9971143534288056,

2.9968783934620418,

2.996633101346655,

2.996378467921658,

2.996114483667674,

2.99584113870551,

2.9955584227946637,

2.995266325331775,

2.9949648353490255,

2.994653941512473,

2.9943336321203318,

2.994003895101194,

2.9936647180121865,

2.993316088037076,

2.992957991984303,

2.9925904162849633,

2.992213346990723,

2.9918267697716683,

2.9914306699140982,

2.991025032318247,

2.9906098414959437,

2.9901850815682067,

2.989750736262771,

2.989306788911546,

2.9888532224480073,

2.9883900194045196,

2.987917161909586,

2.9874346316850313,

2.9869424100431075,

2.9864404778835314,

2.985928815690443,

2.9854074035292957,

2.9848762210436615,

2.9843352474519667,

2.9837844615441447,

2.9832238416782086,

2.982653365776748,

2.982073011323337,

2.981482755358864,

2.9808825744777727,

2.9802724448242217,

2.9796523420881518,

2.979022241501267,

2.9783821178329264,

2.977731945385941,

2.9770716979922778,

2.9764013490086727,

2.9757208713121415,

2.975030237295396,

2.974329418862159,

2.9736183874223774,

2.9728971138873326,

2.972165568664644,

2.9714237216531667,

2.9706715422377785,

2.969908999284058,

2.9691360611328452,

2.968352695594691,

2.967558869944186,

2.9667545509141706,

2.965939704689824,

2.965114296902626,

2.964278292624193,

2.963431656359987,

2.9625743520428873,

2.9617063430266333,

2.960827592079126,

2.959938061375591,

2.9590377124915985,

2.958126506395938,

2.9572044034433422,

2.9562713633670605,

2.955327345271279,

2.9543723076233794,

2.953406208246036,

2.952429004309152,

2.9514406523216223,

2.9504411081229267,

2.949430326874547,

2.9484082630512036,

2.9473748704319105,

2.9463301020908395,

2.9452739103879955,

2.9442062469596944,

2.943127062708841,

2.9420363077950022,

2.940933931624271,

2.9398198828389135,

2.938694109306803,

2.9375565581106233,

2.936407175536845,

2.9352459070644703,

2.9340726973535327,

2.932887490233355,

2.931690228690553,

2.930480854856785,

2.929259309996235,

2.9280255344928245,

2.926779467837155,

2.92552104861316,

2.9242502144844704,

2.922966902180484,

2.9216710474821284,

2.920362585207314,

2.919041449196068,

2.9177075722953374,

2.9163608863434622,

2.9150013221543007,

2.9136288095010014,

2.912243277099418,

2.9108446525911496,

2.909432862526199,

2.908007832345248,

2.906569486361525,

2.905117747742266,

2.903652538489753,

2.9021737794219202,

2.900681390152517,

2.899175289070814,

2.8976553933208415,

2.8961216187801493,

2.894573880038072,

2.8930120903734857,

2.8914361617320496,

2.8898460047029064,

2.8882415284948393,

2.8866226409118596,

2.884989248328219,

2.8833412556628195,

2.8816785663530173,

2.88000108232779,

2.87830870398026,

2.8766013301395486,

2.8748788580419453,

2.873141183301373,

2.8713881998791244,

2.8696198000528543,

2.8678358743848014,

2.8660363116892205,

2.864220998998998,

2.8623898215314307,

2.8605426626531383,

2.8586794038440897,

2.8567999246607094,

2.854904102698043,

2.8529918135509447,

2.8510629307742708,

2.8491173258420295,

2.8471548681054704,

2.84517542475008,

2.8431788607514314,

2.841165038829881,

2.839133819404046,

2.837085060543048,

2.835018617917468,

2.8329343447489777,

2.830832091758604,

2.8287117071135808,

2.8265730363727446,

2.8244159224304206,

2.8222402054587583,

2.8200457228484557,

2.817832309147824,

2.8155997960001375,

2.8133480120792043,

2.811076783023109,

2.80878593136605,

2.80647527646822,

2.8041446344436562,

2.801793818085982,

2.7994226367919834,

2.7970308964829287,

2.7946183995235567,

2.792184944638658,

2.7897303268271467,

2.7872543372735485,

2.784756763256803,

2.782237388056282,

2.7796959908549277,

2.777132346639395,

2.774546226097097,

2.7719373955100304,

2.769305616645263,

2.7666506466419523,

2.7639722378947766,

2.761270137933617,

2.7585440892993742,

2.755793829415746,

2.753019090456819,

2.7502195992103124,

2.747395076936291,

2.744545239221176,

2.741669795826864,

2.7387684505347467,

2.7358409009844404,

2.732886838506992,

2.729905947952345,

2.7268979075108204,

2.7238623885283655,

2.7207990553153096,

2.7177075649483444,

2.7145875670654473,

2.7114387036534358,

2.708260608827845,

2.7050529086047805,

2.7018152206644004,

2.6985471541056567,

2.6952483091918986,

2.6919182770869394,

2.6885566395811376,

2.6851629688070506,

2.681736826944181,

2.678277765912291,

2.6747853270527813,

2.6712590407975396,

2.6676984263246912,

2.664102991200611,

2.660472231007528,

2.6568056289560444,

2.6531026554818027,

2.6493627678255374,

2.6455854095956743,

2.6417700103125976,

2.637915984933666,

2.634022733357984,

2.630089639909885,

2.626116072800027,

2.6221013835629208,

2.6180449064696405,

2.6139459579143978,

2.6098038357735702,

2.6056178187356767,

2.6013871656007295,

2.5971111145472348,

2.592788882365062,

2.5884196636522483,

2.5840026299736674,

2.579536928979406,

2.5750216834804665,

2.570455990479334,

2.5658389201527116,

2.561169514783565,

2.5564467876394352,

2.551669721793716,

2.5468372688864034,

2.541948347820534,

2.5370018433902697,

2.5319966048362916,

2.5269314443238167,

2.5218051353382216,

2.516616410992858,

2.5113639622432347,

2.5060464360012698,

2.500662433142839,

2.4952105064012846,

2.4896891581389604,

2.484096837988256,

2.4784319403527877,

2.472692801758725,

2.466877698045316,

2.4609848413827486,

2.4550123771044836,

2.448958380339993,

2.442820852432661,

2.436597717126161,

2.4302868165011184,

2.423885906642196,

2.4173926530138257,

2.410804625520781,

2.4041192932274393,

2.39733401870705,

2.3904460519894553,

2.383452524072523,

2.376350439959003,

2.3691366711765354,

2.3618079477340856,

2.354360849463053,

2.346791796685718,

2.3390970401473017,

2.331272650140805,

2.3233145047456913,

2.3152182770923133,

2.306979421553629,

2.298593158753914,

2.2900544592707,

2.2813580258908726,

2.272498274264171,

2.263469311777236,

2.254264914448101,

2.244878501614276,

2.23530310815668,

2.225531353965807,

2.2155554103148996,

2.2053669627563766,

2.194957170101014,

2.1843166189728946,

2.1734352733548525,

2.1623024184468225,

2.150906598050054,

2.1392355445601314,

2.127276100496432,

2.1150141303097403,

2.1024344209858286,

2.0895205696923385,

2.0762548563878767,

2.062618098911409,

2.0485894875787327,

2.034146395706845,

2.019264161735797,

2.0039158376807293,

1.988071897470526,

1.9716998972434765,

1.954764077778986,

1.9372248968193353,

1.9190384759009873,

1.9001559422285295,

1.8805226407481122,

1.8600771844340889,

1.8387503012112771,

1.8164634229093821,

1.7931269437370645,

1.7686380508094943,

1.7428779939860515,

1.7157086116057998,

1.6869678546541647,

1.6564639415663487,

1.6239676079701055,

1.5892016540192955,

1.5518265732343897,

1.511420356182263,

1.4674493836742524,

1.419225231444679,

1.3658383175828144,

1.3060516861229083,

1.2381222124142595,

1.1594801576777876,

1.0661065897155806,

0.9511855613380183,

0.8016996663465056,

0.5874683880577859,

0.20470156589570482,

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


us = [u]
uex = [uexact]
uerror = [abs((u - uexact)/uexact)*1000]
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
    deltat= t-tdata[plotnum-1]

    v0 = 0
    for i in range(len(v0part)):
        v0 = v0 + v0part[i]
    gamma=1/numpy.sqrt(1-v0**2)
    tdata[plotnum] = t
    v0part.append( Fana[plotnum] * deltat)

    uexact = -numpy.tanh(gamma * (xx + xt[plotnum])) + numpy.tanh(gamma * (xx - xt[plotnum])) + 1

    uex.extend([uexact])
    uerror.extend([abs((u - uexact)/uexact)*1000])
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
plt.ylabel('$|\\frac{\phi_{simulated}-\phi_{exact}}{\phi_{exact}}|\,\,\,\,\,[10^{-3}]$')
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


