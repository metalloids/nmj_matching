#!/usr/bin/env python3

"""
    A neural algorithm for computing bipartite matchings.
    (c) 2024 Saket Navlakha, navlakha@cshl.edu
"""

# so no GDK warning issues.
import matplotlib
matplotlib.use('Agg')

from optparse import OptionParser
from matplotlib import pylab as plt
from matplotlib import rcParams
import scipy.optimize,scipy.io
import numpy as np
import seaborn as sns
import pandas as pd
import time,math

np.random.seed(10301949)


# Set plotting style.
plt.style.use('fivethirtyeight')
plt.rc('font', **{'sans-serif':'Arial', 'family':'sans-serif'})
rcParams.update({'font.size': 18})


# Parameters set by command line.
N=-1           # number of motor neurons
M=-1           # number of muscle fibers
alpha=-1       # competition rate
beta=-1        # re-allocation rate
degree_dist=-1 # degree distribution
weight_dist=-1 # weight distribution
n_steps=-1     # number of steps per iteration.
n_iters=-1     # number of iterations to average over. 
f=-1           # vector of neuron firing rates. 
plot=False
alloc_type=-1  # proportional or constant re-allocation.

# Constants.
R=100          # resource budget


#==============================================================================
#                           INITIALIZATION FUNCTIONS
#==============================================================================
def normalize(A,add_noise=False):
    """ Normalizes A so the sum of each row (neuron's allocations) is R. 
            @param A -- bipartite matrix
            @param add_noise -- whether to add epsilon noise to each entry
     """

    # Add independent noise to each entry. Useful for symmetry breaking, so two
    # neurons (rows) are never equal.
    if add_noise:
        for i in range(N):
            for j in range(M):
                assert A[i,j] >= 0

                # Add 1-5% noise to the entry.
                A[i,j] = A[i,j] + A[i,j] * np.random.uniform(low=0.01,high=0.05)
    
    row_sums = A.sum(axis=1) # get current allocations for each neuron.
    A = A / row_sums[:, np.newaxis] * R # normalize so each row-sum is R.

    return A


def initialize_graph(p=1.0):
    """ Initializes the bipartite graph with N motor neurons and M muscle fibers.
            @param p -- density of the biparite graph.
    """

    assert degree_dist == "complete"

    # Create N x M allocation matrix; each edge exists with probability p.
    A = np.random.binomial(1,p,size=(N,M))

    # Assign each edge a random weight.
    if weight_dist == "uniform": 
        A = A * np.random.uniform(size=(N,M))
    elif weight_dist == "gaussian":
        A = np.random.normal(size=(N,M))
        A = A + abs(np.min(A)) # to ensure no weights are negative.
    elif weight_dist == "lognormal":
        A = A * np.random.lognormal(size=(N,M))
    elif weight_dist == "poisson":
        A = A * np.random.poisson(size=(N,M))
    else:
        assert False          

    # Scale so the sum of each neuron's weights (budget allocations) is R.
    A = normalize(A)

    run_checks(A)

    return A


def run_checks(A):
    """ Run checks to make sure the initial network is legit.
            @param A -- bipartite matrix.
    """

    # Check that each neuron is connected to at least one fiber and has spent its budget.
    for i in range(N):

        if np.sum(A[i,:] > 0) == 0:
            raise RuntimeError("Neuron %i has 0 fiber connections... exiting." %(i))

        if not math.isclose(R,np.sum(A[i,:])): # checks if abs(R-sum(.)) < 1e-9
            raise RuntimeError("Neuron %i has not spent its budget... exiting." %(i))
            
    # Check that each fiber has at least one neuron connecting to it.
    for j in range(M):
        if np.sum(A[:,j] > 0) == 0:
            raise RuntimeError("Fiber %i has 0 neuron connections... exiting." %(j))
                

#==============================================================================
#                              DATA HELPERS
#==============================================================================
def get_degrees(X,axis):
    """ Returns degree distribution of connectome (X) along the given axis.
            @param X -- connectome matrix; assumed shaped as (#fibers, #neurons).
            @param axis -- 0 is columns (motor neurons); 1 is rows (muscle fibers)
    """
    degrees = np.sum(X,axis=axis)

    assert X.shape[0] > X.shape[1] # check more fibers than neurons.
    assert len(degrees) == X.shape[1-axis] # check summed the right axis.

    return degrees


def read_george_network(filename,N,M):
    """ Returns george (AbtBuy) bipartite matrix. 
            @param filename -- filename of the AbtBuy matrix.
            @param N -- # of Abt.com sellers
            @param M -- # of Buy.com sellers
    """

    # Read the csv file.
    A = np.zeros((N,M))
    with open(filename) as f:
        for line in f:
            cols = line.strip().split(",")
            assert len(cols) == 3 

            u,v,weight = int(cols[0]),int(cols[1]),float(cols[2]) # 0,6,0.03426

            assert weight >= 0
            A[u,v] = weight

    # Scale so the sum of each neuron's weights (budget allocations) is R.
    A = normalize(A)

    run_checks(A)

    return A


def read_OR_network(filename,N,M):
    """ Returns OR bipartite matrix. 
            @param filename -- filename of the OR matrix.
            @param N -- # of nodes on the L side.
            @param M -- # of nodes on the R side.

        The optimal solution (to the minimization) is 2239. You can get this by
        starting with:
            A = np.ones((N,M)) + 10000
            A[u-1,v-1] = weight
            and maximize=False down below (or just take negative vals lol).            

        From: http://people.brunel.ac.uk/~mastjjb/jeb/info.html
    """

    # Read the input file.
    A = np.zeros((N,M))
    with open(filename) as f:
        for line_num,line in enumerate(f):
            if line_num == 0: continue # header.
            cols = line.strip().split(" ")
            assert len(cols) == 3

            u,v,weight = int(cols[0]),int(cols[1]),int(cols[2]) # 1 504 29

            assert 0 <= weight <= 100

            # -- Subtract 1 from the indices since they are 1-based.
            # -- 100-weight bc we want max (not min) matching, and 100 is the largest value.
            A[u-1,v-1] = 100-weight 

    # Scale so the sum of each neuron's weights (budget allocations) is R.
    A = normalize(A)

    run_checks(A)

    return A


def read_celegans_network(filename,N,M):
    """ Returns javier (Celegans) bipartite matrix. 
            @param filename -- filename of javier's matrix.
            @param N -- # of neurons in animal 1
            @param M -- # of neurons in animal 2
    """

    A = np.loadtxt(filename,delimiter=",") # read csv using numpy.

    # Make sure all entries are positive.
    for i in range(N):
        for j in range(M):
            assert A[i,j] >= 0

    # Scale so the sum of each neuron's weights (budget allocations) is R.
    A = normalize(A)

    run_checks(A)    

    return A


def read_movielens(filename,N,M):
    """ Returns original movielens bipartite matrix.
            @param filename -- filename of movielens matrix.
            @param N -- # of movies
            @param M -- # of users

        Movies that have lots of high ratings end up being disconnected because their
        weights get normalized and thus each weight is very low; movies with one, even poor,
        rating will end up more likely matched because it's normalized weight is higher.
    """
  
    # Read the input file.
    A = np.zeros((N,M))
    with open(filename) as f:
        for line_num,line in enumerate(f):
            if line_num == 0: continue # header

            cols = line.strip().split(" ")

            assert int(cols[2]) >= 0

            A[int(cols[1])-1,int(cols[0])-1] = int(cols[2])

    # Scale so the sum of each neuron's weights (budget allocations) is R.
    A = normalize(A,add_noise=True)

    run_checks(A)    

    return A


def read_uiuc(filename,N,M):
    """ Returns UIUC papers-reviewing bipartite matrix.
            @param filename -- filename of uiuc matrix.
            @param N -- # of papers
            @param M -- # of reviewers
    """

    A = np.load(filename)

    # Take transpose because it's currently M x N.
    A = np.transpose(A)

    # Convert cosine distance to cosine similarity.
    A = 1 - A

    # Set a couple small values (-2.22e-16) to 0.
    A[A < 0] = 0

    # Delete the one paper (#186) that has all 0s.
    assert np.all(A[:,186]==0)
    A = np.delete(A,186,1)

    # Scale so the sum of each neuron's weights (budget allocations) is R.
    A = normalize(A,add_noise=True)

    run_checks(A)    

    return A


def read_jesse_network(filename,N,M):
    """ Read Jesse co-expression network. 
            @param filename -- filename of jessey2r matrix
            @param N -- # of yeast genes
            @param M -- # of worm genes
    """

    A = np.zeros((N,M))
    with open(filename) as f:
        for line_num,line in enumerate(f):
            if line_num == 0: continue # header
            cols = line.strip().split(",")

            assert len(cols[1:]) == M
            A[line_num-1] = list(map(float,cols[1:]))

            assert np.all(A[line_num-1] >= 0) # check all positive.   

    # Scale so the sum of each neuron's weights (budget allocations) is R.
    A = normalize(A)

    run_checks(A)    

    return A


#==============================================================================
#                          GRAPH HELPER FUNCTIONS
#==============================================================================
def get_neighbors_neuron(A,i):
    """ Returns the neighbors (fiber indices) of neuron i. """
    return np.nonzero(A[i,:])[0]


def get_neighbors_fiber(A,j):
    """ Returns the incoming neighbors (motor neuron indices) of fiber j. """
    return np.nonzero(A[:,j])[0]


def print_graph(A,k,t):
    """ Prints the bipartite graph at iteration k, timestep t. """

    print("=== Iteration %i, Step %i ===" %(k,t))

    # Prints the connections for each fiber:
    #   fiber: neighbors, weights to each neighbor.
    print("FIBERS")
    for j in range(M):

        neighbors = get_neighbors_fiber(A,j)
        print(j,neighbors,A[:,j][neighbors])

    # Prints the connections for each neuron:
    #   neuron: neighbors, weights to each neighbor.
    print("\nNEURONS")
    for i in range(N):

        neighbors = get_neighbors_neuron(A,i)
        print(i,neighbors,A[i,:][neighbors])


def print_matching(A):
    """ Print (post cleaned-up) matching from fiber-side. """

    # Iterate through each fiber, and print its matched neuron (or 'X' if unmatched).
    for j in range(M):
        nbrs = get_neighbors_fiber(A,j)

        if   len(nbrs) == 0: print("%i->X" %(j),end=' ')
        elif len(nbrs) == 1: print("%i->%i" %(j,nbrs[0]),end=' ')
        else: assert False # matching must be cleaned up.

    print() # newline at the end.


#==============================================================================
#                            OPTIMIZATION FUNCTIONS
#==============================================================================
def get_competition(A,i,j):
    """ Returns the competition term of A_ij for neuron i. """
    
    # Competition: sum of weights coming into fiber j, minus weight from neuron i.
    competition = np.sum(f*A[:,j]) - f[i]*A[i,j]

    assert competition >= 0
    return competition


def get_reallocation(A,i,j):
    """ Returns the re-allocation term of A_ij for neuron i. """

    # Compute the total budget spent so far.
    budget_spent = np.sum(A[i,:])

    # Proportional re-allocation.
    if alloc_type == "prop":

        # Divide the budget remaining proportionally.
        reallocation = (R/f[i] - budget_spent) * (A[i,j] / budget_spent)

    # Constant re-allocation.
    elif alloc_type == "const":
    
        # number of neighbors.
        degree = np.count_nonzero(A[i,:])

        # Each remaining synapse gets boosted by the same amount.
        reallocation = (R/f[i] - budget_spent) / degree

    return reallocation


def update_weights(A):
    """ Applies the two equations for all synapses synchronously. """

    # Apply the competition weight update rule for each A_ij.
    A1 = A.copy() # copy current matrix; otherwise won't be synchronous.
    for i in range(N):
        for j in get_neighbors_neuron(A,i):

            assert A[i,j] > 0

            # Compute competition and re-allocation terms using current matrix.
            competition = get_competition(A,i,j)

            # Compute new weight and update; if negative, set to 0 (i.e., prune edge).
            A1[i,j] = max(A[i,j] - alpha*competition,0)


    # Apply the reallocation weight update rule for each A_ij.
    A2 = A1.copy() # copy current matrix; otherwise won't be synchronous.
    for i in range(N):
        # It's fine here to iterate through A1 (instead of A) bc if competition made you zero,
        # re-allocation won't bring you back (i.e., assume no resurrection).
        for j in get_neighbors_neuron(A1,i): 

            assert A1[i,j] > 0

            # Compute competition and re-allocation terms using current matrix.
            reallocation = get_reallocation(A1,i,j)

            # Compute and update new weight.
            A2[i,j] = A1[i,j] + beta*reallocation

        # Make sure all resources are re-allocated (only if beta = 1.0).
        if beta==1.0: assert math.isclose(R,f[i]*np.sum(A2[i,:]))
        
    return A2


#==============================================================================
#                      MATCHING AND CONVERGENCE FUNCTIONS
#==============================================================================
def converged(A):
    """ For 1-to-1: returns True if each neuron/fiber is matched to one fiber/neuron.
        For 1-to-many: returns True if each fiber is matched to one neuron.
    """

    # This case needs to be run for both types of matchings.
    for j in range(M):

        # If number of fiber neighbors does not equal 1, we're not yet done.
        if len(get_neighbors_fiber(A,j)) != 1: return False


    # This case needs to be run for only 1-to-1 matchings.
    if N == M:
        for i in range(N):

            # If number of neuron neighbors does not equal 1, we're not yet done.
            if len(get_neighbors_neuron(A,i)) != 1: return False

    return True


def clean_up(A):
    """ Cleans up the final graph to produce a true matching. 

        For 1-to-1 matchings, there are two cases:
            a) a neuron splits its resources between two fibers; and 
            b) a fiber receives all resources from two neurons. 
        For both cases, this function deletes the weaker of the two (neuron,fiber) pairs.
        
        For 1-to-many matchings, only case b) is required because neurons
        can have multiple neighbors, but each fiber should have only one neighbor.
    """

    # Iterate through each neuron.
    if N == M:
        for i in range(N):

            # Get the neighbors of the neuron.
            i_neighbors = get_neighbors_neuron(A,i)

            # Case: one neuron -> two+ fibers.
            if len(i_neighbors) >= 2: 

                # Get the max-weight neighbor (fiber).
                jmax = np.argmax(A[i,:]) 

                # Delete all but the max-weight neighbor.                
                for j in i_neighbors:
                    if j == jmax: continue
                    A[i,j] = 0 # delete the weak neighbor.


    # Iterate through each fiber.
    for j in range(M):

        # Get the neighbors of the fiber.
        j_neighbors = get_neighbors_fiber(A,j)

        # Case: two+ neurons -> one fiber.
        if len(j_neighbors) >= 2:

            # Get the max-weight neighbor (neuron)
            imax = np.argmax(A[:,j]) 

            # Delete all but the max-weight neighbor.
            for i in j_neighbors:
                if i == imax: continue
                A[i,j] = 0 # delete the weak neighbor.

    return A


def compute_alg_matching(A):
    """ Computes the matching from the neural algorithm. """

    degrees = np.zeros(N) # degree of each neuron.
    row_ind,col_ind = [],[] # stores the winning (neuron,fiber) pairs.
    for j in range(M):
        nbrs = get_neighbors_fiber(A,j)
        
        if len(nbrs) == 1: # matched fiber

            # Store the matched pair.
            col_ind.append(j)
            row_ind.append(nbrs[0])

            # Increment chosen neuron's degree.
            degrees[nbrs[0]] += 1 

        else: # unmatched fibers
            assert len(nbrs) == 0

    return row_ind,col_ind,degrees


def compute_opt_matching(Ainit,repeats):
    """ Returns the optimal matching using the Hungarian algorithm. 
            @param Ainit -- initial bipartite matrix
            @param repeats -- # of repeats per neuron (set to 1 for 1-to-1 matchings)
    """

    # Run Hungarian on the repeated bipartite graph.
    Ax = np.repeat(Ainit, repeats=repeats, axis=0) # make copies of each row (neuron)
    row_ind,col_ind = scipy.optimize.linear_sum_assignment(Ax,maximize=True)
    value = Ax[row_ind, col_ind].sum()

    # Compute degrees of neurons.
    degrees = np.zeros(N)
    fibers = set()
    for i in range(len(row_ind)):

        # If repeats=10, then rows 0-9 = neuron0, rows 10-19 = neuron1, etc.
        xmatch = int(row_ind[i]/repeats) 

        # Increment chosen neuron's degree.
        degrees[xmatch] += 1

        # Make sure fibers are matched only once.
        assert col_ind[i] not in fibers
        fibers.add(col_ind[i])

    # For 1-to-1 matching, check all degrees == 1.
    if N == M: assert np.all(degrees == 1) 

    # Check all fibers are matched.
    assert len(fibers) == M

    return value,degrees


def compute_distributed_greedy_matching(Ainit):
    """ Returns distributed greedy matching.
        -- Pick a random fiber that has > 1 input.
        -- Select one of its inputs with probability proportional to its inverse weight.
        -- Delete it.
    """

    A = Ainit.copy() # copy because we are going to modify Ainit.

    # Generate list of fiber degrees, so we know when a fiber is matched.
    fiber_degrees = np.zeros(M)
    for j in range(M):
        fiber_degrees[j] = len(get_neighbors_fiber(A,j))


    # Start deleting as long as there are unmatched fibers.
    unmatched_fibers = np.arange(M)
    neuron_degrees = np.zeros(N)
    row_ind,col_ind = [],[]
    while len(unmatched_fibers) > 0:

        # Select a random unmatched fiber.
        j = np.random.choice(unmatched_fibers)

        # Get the non-zero neighbors and weights of the chosen fiber.
        xx = A[:,j]
        neighbors = np.nonzero(xx)[0]
        weights = 1/xx[neighbors] # invert the weights, so small weight -> more likely to pick.

        # Select a random neuron proportional to its inverse weight.
        i = np.random.choice(neighbors,p=weights/weights.sum())

        assert A[i,j] > 0 # make sure its a legit neighbor; i.e., its weight > 0.

        # If this is the last neighbor of the fiber, then store the match.
        if fiber_degrees[j] == 1:
            row_ind.append(i)
            col_ind.append(j)

            # Increment the degree of the chosen neighbor.
            neuron_degrees[i] += 1

            # Remove j from the list of unmatched fibers.
            unmatched_fibers = np.delete(unmatched_fibers, np.where(unmatched_fibers==j))

        # Otherwise, delete the synapse.
        else:
            A[i,j] = 0
            fiber_degrees[j] = fiber_degrees[j] - 1 # decrement fiber degree.


    assert np.all(fiber_degrees == 1)

    return row_ind,col_ind,neuron_degrees


def compute_rand_matching(Ainit):
    """ Returns a random matching. """

    # 1-1 case: for each neuron, pick a random (unpicked) fiber.
    # -- We can't return a random permutation of the neurons because that assumes p=1.
    if N == M:

        row_ind,col_ind = [],[]
        degrees = np.zeros(N) # degree of each neuron.
        fibers_used = set() # set of used fibers.

        # Iterate through the neurons in random order.
        for i in np.random.permutation(range(N)):

            # Pick a random (unpicked) neighbor of i.
            for j in np.random.permutation(get_neighbors_neuron(Ainit,i)):
                if j not in fibers_used: 

                    # Store the match.
                    row_ind.append(i)
                    col_ind.append(j)

                    # Add the fiber to the used set.
                    fibers_used.add(j)

                    # Increment the degree of the chosen neighbor.
                    degrees[i] += 1

                    break

        # It's possible that a neuron is unmatched, if all its neighbors were already picked. 
        return row_ind,col_ind,degrees

    # 1-to-many case: each neuron picks a random neighbor fiber, until all fibers are picked. 
    # -- Requires an order for picking neurons.
    # -- All fibers guaranteed to be matched
    # -- Neurons could be disconnected if all its neighbors are already picked.
    if M > N:
        row_ind,col_ind = [],[]
        degrees = np.zeros(N) # degree of each neuron.

        neurons = np.random.permutation(range(N)) # random order of neurons.
        fibers_used = set() # set of picked fibers.
        idx = 0

        while len(fibers_used) < M:

            # Get the next neuron.
            i = neurons[idx % N]

            # Pick a random (unpicked) neighbor of i.
            for j in np.random.permutation(get_neighbors_neuron(Ainit,i)):
                if j not in fibers_used: 

                    # Store the match.
                    row_ind.append(i)
                    col_ind.append(j)

                    # Add the fiber to the used set.
                    fibers_used.add(j)

                    # Increment the degree of the chosen neighbor.
                    degrees[i] += 1

                    break

            idx += 1

        assert len(set(col_ind)) == len(col_ind) # check no duplicates in col_ind.
        return row_ind,col_ind,degrees


#==============================================================================
#                                   MAIN
#==============================================================================
def main():

    global N,M,alpha,beta,degree_dist,weight_dist,filename,n_iters,n_steps,f,plot,alloc_type

    #========================================================
    start = time.time()

    usage="usage: %prog [options]"
    parser = OptionParser(usage=usage)
    
    parser.add_option("-N", "--neurons", action="store", type=int, dest="N", default=-1,help="number of neurons")
    parser.add_option("-M", "--fibers", action="store", type=int, dest="M", default=-1,help="number of fibers")
    parser.add_option("-a", "--alpha", action="store", type=float, dest="alpha", default=0.001,help="value of alpha")    
    parser.add_option("-b", "--beta", action="store", type=float, dest="beta", default=1.0,help="value of beta")        
    parser.add_option("-d", "--degree", action="store", type=str, dest="degree_dist", default="complete",help="degree distribution [complete,neural]")
    parser.add_option("-w", "--weights", action="store", type=str, dest="weight_dist", default="",help="weights [uniform,zipf]")
    parser.add_option("-f", "--filename", action="store", type=str, dest="filename", default=None,help="filename for input network")
    parser.add_option("-p", "--plot", action="store_true", dest="plot", default=False,help="make summary plots?")

    # Get command line parameters.
    (options, args) = parser.parse_args()

    # Generate random network.
    if options.filename == None:
        filename = None
        N = options.N
        M = options.M
        f = np.ones(N)
        n_iters = 10
        degree_dist = options.degree_dist
        weight_dist = options.weight_dist
        

    # Read Abt-Buy network.
    elif options.filename == "abtbuy":
        filename = "../data/er/Abt-Buy_CHARACTER_BIGRAMS_TF_IDF_COSINE_SIMILARITY.csv"
        N = 1076
        M = 1076
        f = np.ones(N)
        n_iters = 1

        Ainit = read_george_network(filename,N,M)

    # Read Beasley item network.
    elif options.filename == "assignp5000":
        filename = "../data/or/assignp5000.txt"
        N = 5000
        M = 5000
        f = np.ones(N)
        n_iters = 1
        
        Ainit = read_OR_network(filename,N,M)

    # Read the worm network.
    elif options.filename == "celegans":
        filename = "../data/celegans/matrix_2_3.csv"
        N = 47
        M = 47
        f = np.ones(N)
        n_iters = 1

        Ainit = read_celegans_network(filename,N,M)

    # Read the MovieLens1M network.
    elif options.filename == "movielens":
        filename = "../data/movielens/out.movielens-1m" # Original dataset.
        N = 3706
        M = 6040
        f = np.ones(N)
        n_iters = 1

        Ainit = read_movielens(filename,N,M)        

    # Read UIUC papers-reviewer network.
    elif options.filename == "uiuc":
        filename = "../data/uiuc/W1.npy"
        N = 73
        M = 188
        f = np.ones(N)
        n_iters = 1

        Ainit = read_uiuc(filename,N,M)

    elif options.filename == "jessey2r":
        filename = "../data/jesse/yeast_roundworm_CoexpConsSpec_1tomany.csv"
        N = 1501
        M = 2251
        f = np.ones(N)
        n_iters = 1

        Ainit = read_jesse_network(filename,N,M)

    # Read neural data.
    elif options.filename == "yaron":
        # The newborn has biased degrees and we do not know how those biases relate to f. 
        # Instead, we start with N and M values from the adult, with all-to-all 
        # connections and log-normal weights.
        filename = None
        N = 15
        M = 217
        f = np.arange(1,N+1)
        n_iters = 1
        degree_dist = "complete"
        weight_dist = "lognormal"

    else:
        assert False


    # Set parameters.
    alpha = float(options.alpha)
    alpha_orig = alpha # in case we boost, store the original alpha.

    beta  = float(options.beta)
    n_steps = int(10/alpha) # 0.01 -> 1k, 0.001 -> 10k, 0.0001 -> 100k
    plot = options.plot
    repeats = np.ceil(M/N)
    alloc_type = "prop" if M >= 2*N else "const"

    #========================================================
    # Plotting statistics for neurons and fibers.
    if plot:
        frac_nonzero = np.zeros((n_iters,n_steps)) # fraction of non-zero synapses.
        flip_flops   = np.zeros((n_iters,M))       # # of flip flops per fiber.

    # Efficient vs fairness trade-off for each algorithm.
    opt1_eff,opt1_fair = np.zeros((n_iters)),np.zeros((n_iters))
    opt2_eff,opt2_fair = np.zeros((n_iters)),np.zeros((n_iters))
    alg_eff,alg_fair = np.zeros((n_iters)),np.zeros((n_iters))
    gred_eff,gred_fair = np.zeros((n_iters)),np.zeros((n_iters))
    rand_eff,rand_fair = np.zeros((n_iters)),np.zeros((n_iters))

    steps = np.zeros((n_iters))
    max_steps = -1


    # Run those iterations! 
    for k in range(n_iters):
        
        # Initialize the graph.
        if filename == None: Ainit = initialize_graph()

        A = Ainit.copy()

        alpha = alpha_orig # copy the original alpha, in case of boosting.

        # Run iterations
        for t in range(n_steps):
            # Print the initial graph.
            #if t == 0: print_graph(A,k,t)

            # For constant re-allocation, boost at t=5000 just to speed things up.
            if t == 5000 and alloc_type == "const": 
                alpha = alpha * 10

            # Update the weights.
            At = A  # store the previous weights.     
            A = update_weights(A)

            if plot: 
                # Store fraction of non-zero synapses.
                frac_nonzero[k,t] = np.sum(At > 0) / (N*M)

                # Compute flip-flops.
                ff1 = np.argmax(A,axis=0)  # max input for each fiber now.
                ff2 = np.argmax(At,axis=0) # max input for each fiber before.
                flip_flops[k,:] += [val1 != val2 for val1, val2 in zip(ff1, ff2)]

            # Check if converged.
            if converged(A): break # matching.
            if np.array_equal(A,At): break # no change in weights.

        # Print final graph.
        #print_graph(A,k,t+1)
        steps[k] = t+1
        max_steps = max(max_steps,t)

        # Deal with non-perfectly-matched neurons and fibers.
        A = clean_up(A)

        # Compute efficiency and fairness of opt2 matching (OPT, repeats=ceil(M/N))
        value,degrees = compute_opt_matching(Ainit,repeats)
        opt1_eff[k] = value*100
        opt1_fair[k] = np.count_nonzero(degrees)/N*100

        # Compute efficiency and fairness of opt1 matching (OPT-Eff, repeats=M).
        # -- Only for 1-to-many random networks.
        if options.filename == None and M > N:

            # Too much memory to set repeats=1000+, so set to 500 manually.
            if M >= 1000: value,degrees = compute_opt_matching(Ainit,500)
            else: value,degrees = compute_opt_matching(Ainit,M)

            opt2_eff[k] = value*100
            opt2_fair[k] = np.count_nonzero(degrees)/N*100

            # Normalize to opt2.
            value_opt = opt2_eff[k] / 100
            opt2_eff[k] = opt2_eff[k] / value_opt
            opt1_eff[k] = opt1_eff[k] / value_opt

        else:

            # Normalize to opt1.
            value_opt = opt1_eff[k] / 100
            opt1_eff[k] = opt1_eff[k] / value_opt

        # Compute efficiency and fairness of algorithm matching.
        row_ind,col_ind,alg_degrees = compute_alg_matching(A)
        alg_eff[k] = Ainit[row_ind, col_ind].sum() / value_opt * 100
        alg_fair[k] = np.count_nonzero(alg_degrees)/N*100
                
        # Compute value of greedy matching.
        row_ind,col_ind,degrees = compute_distributed_greedy_matching(Ainit)
        gred_eff[k] = Ainit[row_ind, col_ind].sum() / value_opt * 100
        gred_fair[k] = np.count_nonzero(degrees)/N*100 # for distributed algorithm.
        
        # Compute value of random matching.
        row_ind,col_ind,degrees = compute_rand_matching(Ainit)
        rand_eff[k] = Ainit[row_ind, col_ind].sum() / value_opt * 100
        rand_fair[k] = np.count_nonzero(degrees)/N*100    
    
    # Output parameters.
    if options.filename == None:
        print("#N=%i\tM=%i\talpha=%.4f\tbeta=%.2f\tdeg=%s\tweights=%s\trepeats=%i\tniters=%i\tnsteps=%i\talloc=%s" %(N,M,alpha,beta,degree_dist,weight_dist,repeats,n_iters,n_steps,alloc_type))
        out_str = degree_dist+"_"+weight_dist
    else:
        print("#N=%i\tM=%i\talpha=%.4f\tbeta=%.2f\tfile=%s\trepeats=%i\tniters=%i\tnsteps=%i\talloc=%s" %(N,M,alpha,beta,options.filename,repeats,n_iters,n_steps,alloc_type))
        out_str = options.filename

    
    # Output efficiency vs fairness trade-off.
    print("#Efficiency vs Fairness:")

    if options.filename == None and M > N:     
        print("OPT (r=%i)\t%.2f ± %.2f\t%.2f ± %.2f" %(M,np.mean(opt2_eff),np.std(opt2_eff),np.mean(opt2_fair),np.std(opt2_fair)))

    print("OPT (r=%i)\t%.2f ± %.2f\t%.2f ± %.2f" %(repeats,np.mean(opt1_eff),np.std(opt1_eff),np.mean(opt1_fair),np.std(opt1_fair)))
            
    print("Neural Alg.\t%.2f ± %.2f\t%.2f ± %.2f" %(np.mean(alg_eff),np.std(alg_eff),np.mean(alg_fair),np.std(alg_fair)))

    print("Greedy Alg.\t%.2f ± %.2f\t%.2f ± %.2f" %(np.mean(gred_eff),np.std(gred_eff),np.mean(gred_fair),np.std(gred_fair)))

    print("Rand Alg.\t%.2f ± %.2f\t%.2f ± %.2f" %(np.mean(rand_eff),np.std(rand_eff),np.mean(rand_fair),np.std(rand_fair)))        

    print("#Done:\t%.1f steps\t %.2f mins" %(np.mean(steps),(time.time()-start)/60))


    # Make summary plot:
    if plot:
        
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(13.3,9.6))
        fig.tight_layout(h_pad=2.5,w_pad=2.5) # h_pad adds more space between rows.
        
        # -- (A) motor neuron degree distribution.
        if options.filename == "yaron": #plot yaron motor unit sizes, as well.
            
            adult_mat = scipy.io.loadmat('../data/yaron/InterscutularisConnectome_Harvard4CSHL_P60_YFP1_YFP2.mat')
            YFP1 = adult_mat['connectome_YFP1']
            YFP2 = adult_mat['connectome_YFP2']
            adult_neuron = np.concatenate( (get_degrees(YFP1,axis=0),get_degrees(YFP2,axis=0) ))
            
            weights1 = np.ones_like(alg_degrees) / len(alg_degrees)
            weights2 = np.ones_like(adult_neuron) / len(adult_neuron)

            bins=[1,7,14,21,28,36]

            (n,bins,patches) = axes[0,0].hist([alg_degrees,adult_neuron],weights=[weights1,weights2],bins=bins,alpha=0.75,label=["Algorithm","Data"])

            axes[0,0].set_xticks([4,10,17.5,24.5,32])
            axes[0,0].set_xticklabels(["1-6","7-13","14-20","21-27","28-36"])
            axes[0,0].legend()

        else:
            sns.histplot(ax=axes[0,0],stat='probability',data=alg_degrees,discrete=True)

        assert len(alg_degrees) == N # count per neuron.
        axes[0,0].set_ylabel("% of neurons")
        axes[0,0].set_xlabel("Degree")
        axes[0,0].set_ylim(-0.02,1.02)

        # -- (B) correlation between activity and degree.
        m, b = np.polyfit( f, alg_degrees, 1) # compute slope of best-fit line.
        sns.regplot(ax=axes[0,1],x=f,y=alg_degrees,line_kws={'linestyle':'--','linewidth':2.5,'label':"r = %.2f" %(np.corrcoef(f,alg_degrees)[0][1])})
        axes[0,1].legend()
        axes[0,1].set_ylabel("Degree")
        axes[0,1].set_xlabel("Activity ($f$)")
        axes[0,1].set_xlim(-0.02,np.max(f)+1)

        # -- (C) # of flip-flops per fiber.
        df = np.mean(flip_flops,axis=0) # mean along the columns (fibers).
        sns.histplot(ax=axes[1,0],data=df,stat='probability',kde=False,discrete=True)
        assert df.shape[0] == M # count per fiber.
        axes[1,0].set_ylabel("% of fibers")
        axes[1,0].set_xlabel("# of flip-flops")
        axes[1,0].set_ylim(-0.02,1.02)


        # -- (D) fraction of non-zero synapses.
        df = pd.DataFrame(frac_nonzero)
        df_melted = df.reset_index().melt(id_vars='index',var_name='step',value_name='dist')
        sns.lineplot(ax=axes[1,1],data=df_melted,x='step',y='dist')#,label='fraction non-zero')
        axes[1,1].set_ylabel("% of synapses")
        axes[1,1].set_xlabel("Step ($t$)")
        #axes[0,0].set(yscale='log')
        axes[1,1].set_xlim(-10,max_steps-1)
        #axes[0,1].ticklabel_format(style='plain')
        
        # Save plot.
        out_file = "../figs/%s_n%i_m%i_a%.4f_b%.2f.pdf" %(out_str,N,M,alpha,beta)
        plt.savefig(out_file,bbox_inches='tight')
        plt.close()

    
if __name__ == "__main__":
    main()
