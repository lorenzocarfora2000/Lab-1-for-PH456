import numpy as np
import matplotlib.pyplot as plt

#the following script uses the MT19937 algorithm to generate random numbers.

#the seed is selected between  18265  &   6786579.
seed = 6786579
rng = np.random.Generator(np.random.MT19937(seed))
function = "MT19937"

### TASK 1 ####

#number of generated points
N = 10000   
         
sequence = rng.random(N)

m = 100    #number of "bins"
E = N/m    #the expected bin population for a uniform distribution

#graphical check using a histogram:
#m-1 so to get the number of bins to be exactly m
counts, bins, bars = plt.hist(sequence, bins = (m-1))

plt.figure(1)
bars
plt.title("Uniform distribution using "+function+", seed = {}".format(seed))
plt.xlabel("considered range")
plt.ylabel("bin population")
plt.plot(sequence, E*np.ones(N), "--")
plt.show()

#deriving chi squared:
chi = 0
for x in counts:
    chi += ((x-E)**2)/E
    
print("chi squared = {} <= {}".format(chi, m))

### sequential correlation graphical check:

shift = 4000           #evaluated shift
n = 1500               #number of considered points from sequence
seq_n = sequence[:n]
seq_shift = sequence[n+shift:2*n+shift]


plt.figure(2)
plt.scatter(seq_n, seq_shift)
plt.xlabel("values of sequence")
plt.ylabel("values of sequence, shifted by {} points".format(shift))
plt.title("sequential correlation graphical check, using {} points, \n {}-points shift, using ".format(n, shift)+
          function+", S = {}".format(seed))
plt.show()

#autocorrelation:
shift_arr = np.linspace(0, 7000, 1000, dtype = int)  #array of evaluated shifts

autocorrelation = np.zeros(len(shift_arr))
for x in range(len(shift_arr)):
    shift = shift_arr[x]
    seq_shift = np.append(sequence[-shift:], sequence[:-shift])
    autocorrelation[x] = np.correlate(sequence, seq_shift)[0]
    
autocorrelation /= max(autocorrelation)   #normalisation    

plt.figure(3)
plt.plot(shift_arr, autocorrelation)
plt.title("Normalised Autocorrelation using "+function+", S = {}".format(seed))
plt.xlabel("imposed shift")
plt.ylabel("Autocorrelation")
plt.grid()

#excludes 0 shift, as it does not give any information:
autocorr_shift = autocorrelation[1:] 

#study the variation of the "static" part:
mean = np.mean(autocorr_shift)
err = np.std(autocorr_shift)/np.sqrt(len(autocorr_shift))

print("autocorrelation for the sequence = {} +- {}".format(mean, err))

## TASK 2-3 ###

#picking the desired generator, commenting out the unwanted one:

#rand = np.random.Generator(np.random.PCG64())      
def box_test(N0, run, Pl, Pr):
    
    """
    assume all N0 particles start on one side (in our case, left)
    1 means particle in left side
    0 means particle not in left side (right side)
    
    all particles, set as 1 (all starting in the left side):
    """
    
    N_arr = np.ones(N0)
    
    #the iterations over time:
    points = np.zeros(run)
    points[0] = sum(N_arr) 
    
    for i in range(1, run):
        moving_particle = rng.integers(0, N0)  
        
        """
        picking a random particle from the totality:
        if the particle is 1 (in the left), it will have a Pl probability 
        of changing side.

        if the particle is 0 (in the right), it will have a Pr probability 
        of changing side.
        
        a random number between 0 and 1 is picked at random, and checked if it
        is smaller than the imposed probability. If it is, the particle will move.
        """
        
        if N_arr[moving_particle] == 1. and rng.random()<Pl:  
            N_arr[moving_particle] = 0.  
            
        elif N_arr[moving_particle] == 0. and rng.random()<Pr:
            N_arr[moving_particle] = 1.

        points[i] = sum(N_arr)
        
    return points

N0 = 500
run = 10000

p = 1
N_arr = box_test(N0, run, p, p)

plt.figure(4)
plt.plot(range(run), N_arr, ".-", label = "left-side population, $P_L$ = {}".format(p))
plt.plot(range(run), N0-N_arr, ".-", label = "right-side population $P_R$ = {}".format(p))
plt.title("Box test example solution, using "+function)
plt.xlabel("Iteration")
plt.ylabel("Populations in the partitioned sides")
plt.legend()
plt.grid()

p = 0.2
N_arr = box_test(N0, run, p, p)

plt.figure(5)
plt.plot(range(run), N_arr, ".-", label = "left-side population, $P_L$ = {}".format(p))
plt.plot(range(run), N0-N_arr, ".-", label = "right-side population $P_R$ = {}".format(p))
plt.title("Box test example solution, using "+function)
plt.xlabel("Iteration")
plt.ylabel("Populations in the partitioned sides")
plt.legend()
plt.grid()

## Task 4 ###
pl, pr = 0.75, 0.25
N_arr = box_test(N0, run, pl, pr)

plt.figure(6)
plt.plot(range(run), N_arr, ".-", label = "left-side population, $P_L$ = {}".format(pl))
plt.plot(range(run), N0-N_arr, ".-", label = "right-side population $P_R$ = {}".format(pr))
plt.title("Box test example solution \n using "+function)
plt.xlabel("Iteration")
plt.ylabel("Populations in the partitioned sides")
plt.legend()
plt.grid()

pl, pr = 0.25, 0.75
N_arr = box_test(N0, run, pl, pr)

plt.figure(7)
plt.plot(range(run), N_arr, ".-", label = "left-side population, $P_L$ = {}".format(pl))
plt.plot(range(run), N0-N_arr, ".-", label = "right-side population $P_R$ = {}".format(pr))
plt.title("Box test example solution using "+function)
plt.xlabel("Iteration")
plt.ylabel("Populations in the partitioned sides")
plt.legend()
plt.grid()
