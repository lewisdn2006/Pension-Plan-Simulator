"""
Pension Plan Simulator
======================

Monte Carlo simulation of a pension that is paid into every month and invested
in a share index. The index follows geometric Brownian motion,

    dS = mu * S dt + sigma * S dW

so each simulated path is one possible future for the stock market. Simulating
a large number of paths gives the whole distribution of outcomes instead of a
single forecast, which answers questions like:

    - How likely am I to end up with less than I paid in?
    - How likely am I to double my money, or to retire with over 2 million?
    - How much do I need to pay in each month for a 95% chance of reaching a
      target fund, and how much more does it cost if I start 20 years later?

Run it with:

    python pension_simulator.py

Requires: numpy, matplotlib
"""

import math
import time

import numpy
import matplotlib.pyplot as plot


sigma = 0.15 # annual volatility of the share index
S0 = 1 # starting share price (the final pension value does not depend on this, see pension_value)


# ---------------------------------------------------------------------------
# Simulating the share price
# ---------------------------------------------------------------------------

def simulate_gbm(mu, P, T, N, seed=None):
    """Simulate P paths of geometric Brownian motion over T years in N time steps.

    Returns a matrix with one row per path and N+1 columns (the price at each
    time step, starting at S0).

    Each step uses the exact solution of the equation,
        S(n+1) = S(n) * exp((mu - sigma^2/2) * dt + sigma * dW),
    and not the simpler step S(n+1) = S(n) + mu*S(n)*dt + sigma*S(n)*dW.
    The simpler step can make the price go negative when sigma*S*dW is less
    than -mu*S*dt. The exact step multiplies a positive price by an
    exponential, so the price stays positive on every path.

    Pass a seed to get the same paths every time.
    """
    delta_t = T/N
    matrix = numpy.zeros((P, N+1)) # creating a matrix the right size

    matrix[:, 0] = S0 # all of the paths start at S0 as t=0

    rng = numpy.random.default_rng(seed)
    deltaW = rng.normal(0, math.sqrt(delta_t), size=(P,N)) # generating a lot of random deltaW using normal distribution and storing them in a 2d array to be used later

    for j in range(1,N+1): # going through each time step, starting at 1 because column 0 is filled already with S0

        matrix[:,j]=matrix[:, j-1]*numpy.exp((mu-(sigma**2)/2)*delta_t+sigma*deltaW[:, j-1]) # filling in the next entry in the matrix using the exact solution

    return matrix


# ---------------------------------------------------------------------------
# Valuing the pension
# ---------------------------------------------------------------------------

def pension_value(paths, T, M):
    """Final value of the pension on every path after T years of paying in M a month.

    Each month M buys M/S shares at that month's price S. The final value is
    the total number of shares held multiplied by the final price.

    The result does not depend on the starting price S0: every price on a path
    is S0 times an exponential, so the number of shares bought carries a
    factor of 1/S0 and the final price a factor of S0, and they cancel.
    """
    if M<0:
        raise ValueError("Monthly input is negative. Try again")

    relevant_paths=paths[:, :T*12+1] # creates a new path matrix which only takes the first T*12 values

    H = numpy.sum(M/relevant_paths, axis=1) # number of shares held at the end

    V = H*paths[:, T*12] # value = holdings x final share price

    return V

def observed_probability(paths, M, T, mu, Vmin):
    """Fraction of paths on which the pension finishes above Vmin."""
    values=numpy.array(pension_value(paths, T, M)) # gets the array from the pension values function and turns it into a numpy array

    count_above = len(values[values>Vmin])

    probability = count_above/paths.shape[0]

    return probability


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_paths(paths, T, N):
    """Plot five example paths with the mean and the 5th and 95th percentiles."""
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

def plot_final_values(values, bin_width=50000):
    """Histogram of the final pension values. Returns the fullest bin and how many paths are in it."""
    min_val = numpy.floor(numpy.min(values) / bin_width) * bin_width # rounds the minimum value down to the nearest bin width
    max_val = numpy.ceil(numpy.max(values) / bin_width) * bin_width # rounds the max value up to the nearest bin width
    bins = numpy.arange(min_val, max_val+bin_width, bin_width) # creates an array to go from the min to max value in steps of the bin width

    frequency_array, bin_array, _ = plot.hist(values, bins=bins, color='blue', edgecolor='black', alpha=0.7, linewidth=0.3)
    plot.xlabel('Final Pension Value')
    plot.ylabel('Frequency')
    plot.title('Distribution of pension values at retirement')
    plot.grid(True, linestyle='--', alpha=0.5)
    plot.show()

    index = numpy.argmax(frequency_array) # uses the numpy argmax function to find the index of the bin with the highest frequency
    highest_count = int(frequency_array[index]) # gets the frequency of the bin in that index
    highest_bin = bin_array[index] # gets the bin that has the most paths

    return highest_bin, highest_count

def plot_probability_curves(M_values, probability_matrix, mu_list):
    """Plot the chance of reaching the target against the monthly contribution, one line per growth rate."""
    colours = ['blue', 'red', 'green']

    for i in range(len(mu_list)):
        plot.plot(M_values, probability_matrix[i, :], label='mu='+str(mu_list[i]), color=colours[i % len(colours)], linestyle='-')

    plot.title('Probability of comfortable retirement')
    plot.xlabel('Monthly Input')
    plot.ylabel('Probability of comfortable retirement')
    plot.legend()
    plot.grid(True)
    plot.show()


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def outcome_probabilities(values, T, M):
    """Chance of a loss, of doubling the money paid in, and of finishing above 2 million."""
    values = numpy.array(values)
    P = values.shape[0]

    loss_counter = len(values[values < 12*T*M]) # finished with less than was paid in
    double_counter = len(values[values > 2*12*T*M])
    twomil_counter = len(values[values > 2000000])

    return loss_counter/P, double_counter/P, twomil_counter/P

def probability_curves(T, Vmin, mu_list, M_spacing, M_max, P):
    """Chance of finishing above Vmin for every monthly contribution from 0 up to M_max, for each growth rate.

    Returns the contributions tested and a matrix with one row per growth rate.
    """
    M_values = numpy.arange(0, M_max, M_spacing) # the monthly contributions to test
    probability_matrix = numpy.zeros((len(mu_list), len(M_values))) # creates a matrix of the correct size for the probabilities

    previous_time = time.time()

    for i in range(len(mu_list)):

        mu = mu_list[i]

        paths=simulate_gbm(mu, P, T, 12*T)

        for j in range(len(M_values)):
            M = M_values[j]

            prob = observed_probability(paths, M, T, mu, Vmin)

            probability_matrix[i, j] = prob

        current_time = time.time()
        print('Computed for mu =',mu,'in',current_time-previous_time, 'seconds')
        previous_time = current_time

    return M_values, probability_matrix

def required_contribution(M_values, probabilities, target=0.95):
    """Smallest monthly contribution tested that reaches the target probability (None if none of them do)."""
    for current_row in range(len(M_values)): # goes through each contribution until the probability of comfortable retirement is above the target
        if probabilities[current_row] >= target:
            return M_values[current_row]

    return None


# ---------------------------------------------------------------------------
# Running the whole analysis
# ---------------------------------------------------------------------------

def main():
    start_time = time.time()

    # --- one scenario in detail: 40 years, 5% growth, 1000 a month ---
    mu = 0.05
    P = 100000
    T = 40
    N = 12*40
    M = 1000

    print('Calculating and plotting paths of share prices over 40 years with 5% annual growth')
    paths = simulate_gbm(mu, P, T, N)
    plot_paths(paths, T, N)

    print('Plotting final values of pension paths on a histogram')
    values = pension_value(paths, T, M)
    highest_bin, highest_count = plot_final_values(values)
    print("The bin with the most paths in it is", highest_bin, "with", highest_count, "paths\n")

    print('Calculating probabilities of having certain amounts in the pension fund at the end of 40 years')
    loss, double, twomil = outcome_probabilities(values, T, M)
    print('probability of loss:', loss)
    print('probability of doubling', double)
    print('probability of over 2 million', twomil)

    # --- how much do you need to pay in for a 95% chance of a 1 million fund? ---
    Vmin = 1000000
    mu_list = [0.03, 0.05, 0.07]
    M_spacing = 20

    needed = {} # needed[T][mu] = monthly contribution

    for T, M_max in [(40, 4000), (20, 10000)]: # starting 40 years before retirement, and starting 20 years before
        print('Calculating probability of comfortable retirement with monthly investments from 0 to', M_max, 'over', T, 'years')

        M_values, probability_matrix = probability_curves(T, Vmin, mu_list, M_spacing, M_max, 100000)
        plot_probability_curves(M_values, probability_matrix, mu_list)

        needed[T] = {}

        for i in range(len(mu_list)):
            mu = mu_list[i]
            amount = required_contribution(M_values, probability_matrix[i, :])
            needed[T][mu] = amount

            if amount is None:
                print('For mu =', mu, 'no contribution up to', M_max, 'gives a 95% chance of comfortable retirement')
            else:
                print('For mu =', mu, 'for a 95% chance of comfortable retirement you need to put in £'+str(amount), 'per month')

    for mu in mu_list:
        print("The amount needed for when mu =", mu,"is", needed[40][mu],"for when T=40 and", needed[20][mu],"for when T=20")

        if needed[40][mu] is not None and needed[20][mu] is not None:
            print("The difference in these two values is", needed[20][mu]-needed[40][mu])

    print('Total time:', time.time()-start_time, 'seconds')


if __name__ == "__main__":
    main()
