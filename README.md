# kinksnumerics
numerical simulation of kink solution

In this repository there are several programs shown that simulate numerically dynamic kink solutions of the phi^4-theory and the Sine-Gordon model, however, an arbitrary potential for a real 1 dimensional scalar can be introduced (Further details below) and numerically simulated.
For simplicity, every model has its own python file. We even split one and mulit-kink solution for the phi^4-model into separate files which would not be necessary. 
The files Kkequ_Kink_commented.py and  Kkequ_two_Kink.py simulate the phi^4-theory, Kkequ_two_kink.py the Sine-Gordon model.

The plots that are included in this repository are labeled such that there is no need of further explanation of what they show. They are all produced by the python files in this repository.IMPORTANT: We always plit simulation and analytical result over each other. Simulations in the gif-files are ALWAYS printed red while the analytical solution is printed blue!
Note however that some plots for static solutions or potentials (since they are trivial analytic functions) are simply done by an simple plot script which is not included in this repository.

The Program Structure: 
We use in implicit-explicit time scheme stepping  applying a degree-three-discretization which follows closely the description of 
https://en.wikibooks.org/wiki/Parallel_Spectral_Numerical_Methods/The_Klein-Gordon_Equation.
Note however, that the equations of motions have to be adapted depending on the potential. For theoretical explanations of how the program works as well as specific instructions on how to modify it please read ….pdf of this repository. A short summary is given in the following:

The program Kkequ_Kink_commented.py is sufficiently commented and can be downloaded and modified as one wish. The other programs include specific calculations for the problems of phi^4 and Sine-Gordon and can be ignored further. Following the Steps in the comments of  Kkequ_Kink_commented.py: 

Step 0: Here, we define intervals in which space and time should be discretized as well as the maximal simulation time and the initial velocity of the kink solution. These can be adapted as it suits.

Step 1: The lines marked with step 1 are supposed to be filled with the analytical solution for the dynamic kink. For multi-kink  solutions just put in the whole solution without any distinction between kinks or antikinks. Note that there are two functions to be filled where “uold” corresponds to the analytical solution one time step earlier. There is a third function “unew” defined with all the other arrays. This is exactly what we want to calculate in each time step. These three functions correspond to our degree-three-discretization (further information see in regard to the latter to statements can be found in ...pdf).

Step 2: This is the part that must be adapt depending on the potential term of the theory. Here the expressions for the different types of energy densities in terms of the field should be implemented.

After step 2 there are a lot of lines dedicated to the analytic calculation of the energies (if there is such). Naturally, these can be calculated in case of a exact analytical solution for the kink dynamics as in the case of the one moving kink in phi^4 or the two kink solution in Sine-Gordon. 

Step 3: In the loops that run the time evolution one can find step 3 where the Fourier transformed equation of motion has to be entered solved for the Fourier transform of the field at the next time step. A more elaborated description will be found in ….pdf. 

Step 4: Once again the exact solution has to be entered.

After this step the simulation is good to go. The animation segments that follow have to be adapted for individual purposes. Note however that the time evolution of the functions are saved in the arrays defined in line 142-148. Since they are distinctly different from the time evolution arrays for the energies defined in line 127-139 different animation procedures are defined. They, too, can be adapted as one wish.

Remark: The interaction force is calculated in KKequ_two_kink.py and corresponding brief comments can be found in the file. Further technical explanations can be found in ....pdf.

Remark: In some plots there is an error calculated as a normalized difference between what is called $\phi_simulated$ and $\phi_exact$. Note that $\phi_exact$ is actually just the analytical solution or an approxiamtion based on asymptotic behavior (no numerics involved) of the corresponding problem. In the case of kink-antikink interaction in the $\phi^4$-theory e.g. there is no analytic solution describing the scattering, therefore, we derive an asymptotic analytic solution based on the interaction force (see ...pdf). This is, however, still labeled as $\phi_exact$.

