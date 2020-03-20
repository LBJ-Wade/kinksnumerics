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
