# kinksnumerics
numerical simulation of kink solution

Note: the Kink.py is exactly the program from https://jakevdp.github.io/blog/2012/09/05/quantum-python/. KKequ_Kink.py is the 
ansatz from https://en.wikibooks.org/wiki/Parallel_Spectral_Numerical_Methods/The_Klein-Gordon_Equation which I modified in 
order to simulate the kink-antikink-annihilation for lambda-phi^4-theory. 

So far:
As stated before the reference program on the stated website was modified in order to simulate the KKequ of the theory of
interest. This was achieved with only a few steps. At first, the function phi and its fourier transform were added into the program.
Then the fourier transform of the equation of motion was added into the simulation loop which loops over and defined number of timesteps
of adjustable length.
[The construction details of the fouriertransform for the simulation will be stated in the report at the end. I have handwritten
notes about that but typing this into this readme seems a bit senceless.] 
I safed the numerically simulated function phi at a given time step in an array of functions and animated this array. 
The result will be uploaded soon, I will also cut out some pictures for the report. 

TODO:
a) For a single kink solution, just show that the analytic solution of one *single* moving kink reproduces your numerical one.

b) Compute and display in "live" quantities like the total energy of the system and it's break-down between kinetic and potential, computed directly from the numerical solution. Compare against analytical.

c) For two-kink solutions: Let the numerical solution evolve from two *initially* static kink and anti-kink solution separated by a distance 2a. (Eq. 5.28). Compute the "effective classical" attractive force experienced by the two kinks (again numerically from Eq. 5.26) and compare it to the analytical approximation of Eq. 5.31.
You can then evolve analytically the starting solution of Eq. 5.28 like if they were two point-like objects of mass m subject to the force computed above. Namely: simply changing `a` of Eq. 5.28 into `x(t)` where `x(t)` is then simply `x0 + v_0*t + 1/2 (F/m) t^2` (with `x0=+/-a` and `v_0 = 0` in the suggested case above). This is what I called the "analytical solution". Now of course at the point of contact I would expect the two to significantly differ, but during the initial approach, they should be similar!

