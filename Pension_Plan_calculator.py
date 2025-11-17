###Question C1

# Equation C.2.1 allows the possibility of price becoming negative if sigma*Sn*deltaWn becomes less than -mu*Sn*deltat
# Equation C.2.2 does not allow the price to become negative if the starting price is positive. If S0 is positive it is multiplied by the exp part which is always positive meaning that Sn is positive for all n
# The solution C.1.6 requires S to be positive for all t if S(0) is positive as it is multiplied by an exponential
# Equation C.2.2 comes deriectly from the solution C.1.6

###Question C2

import numpy
import math
import time
import sys


start_time = time.time()

sigma = 0.15
S0 = 1
def simulate_gbm(mu, P, T, N):
    delta_t = T/N
    matrix = numpy.zeros((P, N+1)) # creating a matrix the right size 
    
    matrix[:, 0] = S0 # all of the paths start at S0 as t=0
            
    rng = numpy.random.default_rng()
    deltaW = rng.normal(0, math.sqrt(delta_t), size=(P,N)) # generating a lot of random deltaW using normal distribution and storing them in a 2d array to be used later

    for j in range(1,N+1): # going through each time step, starting at 1 because row 0 is filled already with S0
    
        matrix[:,j]=matrix[:, j-1]*numpy.exp((mu-(sigma**2)/2)*delta_t+sigma*deltaW[:, j-1]) # filling in the next entry in the matrix using the fomula C.2.2
            
    return matrix


###Question C3

print('Calculating and plotting paths of share prices over 40 years with 5% annual growth')

#https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html

import matplotlib.pyplot as plot

# set all of the values that the question set
mu = 0.05
P=100000
T=40
N=12*40

# create the paths 2d array
paths=(simulate_gbm(mu, P, T, N))

#first 5 paths are just the first 5 rows of the paths array
path1=paths[0, :]
path2=paths[1, :]
path3=paths[2, :]
path4=paths[3, :]
path5=paths[4, :]

#mean path using numpy functions
mean_path=numpy.mean(paths, axis=0)

#percentiles
path_5percentile=numpy.percentile(paths, 5, axis=0)
path_95percentile=numpy.percentile(paths, 95, axis=0)

# creating x axis from 0 to T with N+1 evenly spaced points
x = numpy.linspace(0, T, N+1)

# plotting all of the paths with different colours and line styles
plot.plot(x, path1, label='path 1', color='pink', linestyle='--')
plot.plot(x, path2, label='path 2', color='orange', linestyle='--')
plot.plot(x, path3, label='path 3', color='yellow', linestyle='--')
plot.plot(x, path4, label='path 4', color='green', linestyle='--')
plot.plot(x, path5, label='path 5', color='blue', linestyle='--')
plot.plot(x, mean_path, label='mean', color='purple', linestyle='solid')
plot.plot(x, path_5percentile, label='5th percentile', color='black', linestyle='-.')
plot.plot(x, path_95percentile, label='95th percentile', color='red', linestyle='-.')

# create the graph
plot.title('Pension Paths Plotted')
plot.xlabel('Time passed(Years)')
plot.ylabel('Share price')
plot.legend()
plot.grid(True)
plot.show()

###Question C4

def pension_value(paths, T, M):

    if M<0:
        print("Monthly input is negative. Try again")

        sys.exit(0)

    values = [] # create an empty array which will be filled with the values calculated
    
    relevant_paths=paths[:, :T*12+1] # creates a new path matrix which only takes the first T*12 values
    
    H = numpy.sum(M/relevant_paths, axis=1) 
    
    V = H*paths[:, T*12] # uses the value formula given

    values.append(V) # adds the value for this path to the values array

    return values

#print("Testing pension value function with suitable values")
#print(pension_value(paths, 40, -1))


###Question C5

print('Plotting final values of pension paths on a histogram')

M = 1000

values=(pension_value(paths, T, M))


bin_width = 50000 
min_val = numpy.floor(numpy.min(values) / bin_width) * bin_width # rounds the minimum value down to the nearest 50000
max_val = numpy.ceil(numpy.max(values) / bin_width) * bin_width # rounds the max value up to the nearest 50000
bins = numpy.arange(min_val, max_val+bin_width, bin_width) # creates an array to go from the min to max value in steps of 50000

frequency_array, bin_array, _ = plot.hist(values, bins=bins, color='blue', edgecolor='black', alpha=0.7, linewidth=0.3)
plot.xlabel('Final Pension Value')
plot.ylabel('Frequency')
plot.title('Distribution of pension values at retirement')
plot.grid(True, linestyle='--', alpha=0.5)
plot.show()

index = numpy.argmax(frequency_array) # uses the numpy argmax function to find the index of the bin with the highest frequency
highest_count = int(frequency_array[index]) # gets the frequency of the bin in that index
highest_bin = bin_array[index] # gets the bin that has the most paths
print("The bin with the most paths in it is", highest_bin, "with", highest_count, "paths\n")

###Question C6

# the final value has the same distribution as the random normal distribution keeps the same mean and variance and the monthly deposits stay the same
# We are given that the final value V(T)=H(T)S(T) where H(T) is the number of holdings at T and S(T) is given by S(T)=S(0)*exp((mu-sigma^2/2)T+sigmaW(n))
# H(T) is equal to the sum from n=0 to T of the monthly input divided by the S(n). This gives the number of shares bought each month and sums it together
# S(n) in this equation can be rewritten using the formula before giving S(0)*exp(...)
# Thus V(T) can be written as V(T)=(sum from n=1 to T of (M/S(0)*exp(...)))*(S(0)*exp(...)) and by linearity of sums the 1/S(0)can be brought out and cancels with the S(0) on the outside
# This leaves V(T) independent of S(0) which is the starting price
# Therefore the final value of the pension is independent of the starting price

###Question C7

print('Calculating probabilities of having certain amounts in the pension fund at the end of 40 years')

values = numpy.array(values)

loss_counter = len(values[values < 12*T*M])
double_counter = len(values[values > 2*12*T*M]) 
twomil_counter = len(values[values > 2000000])

print('probability of loss:', loss_counter/paths.shape[0])
print('probability of doubling', double_counter/paths.shape[0])
print('probability of over 2 million', twomil_counter/paths.shape[0])

###Question C8

print('Calculating probability of comfortable retirement with monthly investments from 0 to 4000 over 40 years')

def observed_probability(paths, M, T, mu, Vmin):
    
    values=numpy.array(pension_value(paths, T, M)) # gets the array from the pension values function and turns it into a numpy array

    count_above = len(values[values>Vmin])

    probability = count_above/paths.shape[0]

    return probability

# setting values for the functions
T = 40
Vmin = 1000000
mu_list = [0.03, 0.05, 0.07]
M_spacing = 20
M_max = 4000

probability_matrix = numpy.zeros((len(mu_list), int(M_max/M_spacing))) # creates a matrix of the correct size for the probabilities 
M_matrix = numpy.zeros((len(mu_list),2))

previous_time = time.time()

for i in range(len(mu_list)):

    mu = mu_list[i]
    
    paths=simulate_gbm(mu, 100000, T, 12*T)

    for j in range(0, int(M_max/M_spacing)):
        M = j*M_spacing

        prob = observed_probability(paths, M, T, mu, Vmin)

        probability_matrix[i, j] = prob

    current_time = time.time()
    print('Computed for mu =',mu,'in',current_time-previous_time, 'seconds')
    previous_time = current_time

x = numpy.linspace(0, M_max, int(M_max/M_spacing)) # creates the x axis for the graph

plot.plot(x, probability_matrix[0, :], label='mu=0.03', color='blue', linestyle='-')
plot.plot(x, probability_matrix[1, :], label='mu=0.05', color='red', linestyle='-')
plot.plot(x, probability_matrix[2, :], label='mu=0.07', color='green', linestyle='-')
plot.title('Probability of comfortable retirement')
plot.xlabel('Monthly Input')
plot.ylabel('Probability of comfortable retirement')
plot.legend()
plot.grid(True)
plot.show()

for i in range(len(mu_list)):
    mu = mu_list[i]

    current_row = 0
    current_probability = 0

    while current_probability < 0.95: # goes through each row until the probability of comfortable retirement is above 95%
        current_probability = probability_matrix[i, current_row]
        current_row += 1

    print('For mu =', mu, 'for a 95% chance of comfortable retirement you need to put in £'+str(current_row*M_spacing), 'per month')
    M_matrix[i, 0] = current_row*M_spacing

###Question C9

print('Calculating probability of comfortable retirement with monthly investments from 0 to 10000 over 20 years')

# resetting some values for this question
T = 20
M_max = 10000

probability_matrix = numpy.zeros((len(mu_list), int(M_max/M_spacing)))

previous_time = time.time()

for i in range(len(mu_list)):

    mu = mu_list[i]
    
    paths=simulate_gbm(mu, 100000, T, 12*T)

    for j in range(0, int(M_max/M_spacing)):
        M = j*M_spacing

        prob = observed_probability(paths, M, T, mu, Vmin)

        probability_matrix[i, j] = prob

    current_time = time.time()
    print('Computed for mu =',mu,'in',current_time-previous_time, 'seconds')
    previous_time = current_time

current_time = time.time()
print(current_time-start_time)

x = numpy.linspace(0, M_max, int(M_max/M_spacing))

plot.plot(x, probability_matrix[0, :], label='mu=0.03', color='blue', linestyle='-')
plot.plot(x, probability_matrix[1, :], label='mu=0.05', color='red', linestyle='-')
plot.plot(x, probability_matrix[2, :], label='mu=0.07', color='green', linestyle='-')
plot.title('Probability of comfortable retirement')
plot.xlabel('Monthly Input')
plot.ylabel('Probability of comfortable retirement')
plot.legend()
plot.grid(True)
plot.show()

for i in range(len(mu_list)):
    mu = mu_list[i]

    current_row = 0
    current_probability = 0

    while current_probability < 0.95:
        current_probability = probability_matrix[i, current_row]
        current_row += 1
    
    print('For mu =', mu, 'for a 95% chance of comfortable retirement you need to put in £'+str(current_row*M_spacing), 'per month')
    M_matrix[i, 1] = current_row*M_spacing

for i in range(len(mu_list)):
    print("The amount needed for when mu =", mu_list[i],"is", M_matrix[i,0],"for when T=40 and", M_matrix[i,1],"for when T=20")
    print("The difference in these two values is", M_matrix[i,1]-M_matrix[i,0])