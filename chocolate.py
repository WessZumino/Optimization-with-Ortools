"""
Chocolate distribution problem (or how to make sure young children will get their cavities)
============================================================================================
Author: Mirco Milletari <milletari@gmail.com>

Statement:
There are m chocolate bars of varying (integer) length and n hungry children who want differing amounts of chocolate
(again integer amounts). You can cut the chocolate bars and the goal is to ensure that every child gets the amount they want.
Write a program to distribute the chocolate so that you make the minimum number of cuts.

Input:
B --- array whose elements contain the number of units available in each bar. Its length -m- correponds to the number of bars.
C---  array whose elements contain the number of units requested by each children. Its length -n- correponds to the number of
      children.

Returns:
Assigned chocolate units such that the number of cuts is minimized

Solution Strategy: The problem is mapped to a minimum cost flow problem. The cost of each "distribution channel" is given by the
"distance" between available and required (chocolate) units: cost[i,j]= |B[i]-C[j]|. In this way, assignements of resources that do not require a cut have 0 cost, so they always minimize the cost
function (e.g. bar: 2 --> child: 2). The rest of the Strategy is to make less costly configurations that are close to each other.

Note: A cut is the number of times we break up the chocolate bar, so in the given example {2,5,7} --> {3,2,5,1}, the minimum cut is one instead of two as we have

Bar 1  assigned 2 units to child  2
Bar 2  assigned 5 units to child  3
Bar 3  assigned 3 units to child  1
Bar 3  assigned 1 units to child  4

So, bar 3 has been cut into two pieces with a single cut

Numerical Implementation:

We use a series of helper functions to break up the problem into different stages. The main program is the min_cut() function.
We use Google's constrained optimization libraries ortools: https://developers.google.com/optimization/

On Unix or MacOs you can install the utitlies using: pip install --upgrade ortools
in the command shell. For other operating system follows the instructions in the link above.

"""
#Import libraries

import numpy as np  #Numpy
from ortools.constraint_solver import pywrapcp #google libraries for constrained optimization
import networkx as nx #graph creation librrary


#----------------------Helper functions-----------------------------------------

# Can every children obtain the requested amount of chocolate? Said Otherwise, is the problem, as stated, satisfiable? The sat()
#function makes sure that the problem can be solved. It also introduces a "ghost" child to ensure the effort/demand neutrality.

def sat(Bars, Children):
    """
    Arguments:
    Bars    -- chocolate bars array
    Children-- children array

    Return:
    f -- flag. f=1 if the problem is satifiable, 0 Otherwise.
    S -- message satisfiable/not satisfiable
    Cp-- if there is an excess amount, create a "ghost" children to take up the excess units
    """

    NB = sum(Bars) #Total number of available chocolate units
    NC = sum(Children) # Total number of requested chocolate units

    Cp= Children #initialize
    ng= len(Children)

    ex= NB-NC #excess units

    if ex == 0:
        S= "The problem is satisfiable"
        f=1

    elif ex > 0:
        Cp = Children+[ex] # ass the "ghost" children if there is an excess supply
        ng= len(Cp) # Lenght of Cp. This is diffrent from -n- if there is a ghost
        S= "The problem is satisfiable"
        f=1

    else :
        S="The problem is not satisfiable" # The problem cannot be satisfied if there is no enough supply
        f=0

    return S, Cp, ng, f


#initialize the paramters of the problem. It implictly generates the computational graph, introducing nodes and vertices.

def initialize(Bars, Children, solver):

    """
    This function initialize the paramters in the problem.

    Arguments:
    Bars--- array of chocolate bars
    Children--- array of children


    Return:
    cost--- (m, ng) matrix; each element contains the cost of each connection. This is obtained by taking the element-wise
            ratio: (supplied[i]-demanded[j]/supplied[i]). If there is an excess is supply a zero weight is assigned to the ghost unit,
            so that it can absorb all the excess supply. Note that the optimized demands integer entries for cost. For this reason
            we multiply the result by 10 and round up to the closer integer using the ceiling function.

    X   --- (m, ng) connection matrix. Matrix elements contain how many units are moved between two nodes of the graph. Here the
            connection marix is initialized.

    bias--- (m+ng) array containg the supplied units (positive) and the demanded ones (negative).

    """

    maxcap = int(max(max(Bars),max(Children))) #abs max number of units on a edge. It is used to set the possible values of the connection matrix X

    ng= len(Children) #Number of children, eventually including the ghost
    bias = Bars+Children #initialize the bias vector as a list

    cost= np.zeros([m,ng]) #initialize the cost matrix

    #initialize the cost matrix
    for i in range(m):
        for j in range(ng):
            #cost[i,j]= int(np.ceil( (abs(Bars[i]-Children[j])/Bars[i])*10))
            cost[i,j]= int( abs(Bars[i]-Children[j]))

    if n != ng: #If there is a ghost, then the cost associated to the connestions leading to it is set zero.
       cost[:,-1] = 0

    cost= list(cost) # The cost matrix is turned into a list in order to work with the ortools optimizer

    #initialize the connetion matrix. Note that while x is a list tyoe, x_M is am array type

    X = []
    for i in range(m):
        t = []
        for j in range(ng):
            t.append(solver.IntVar(0, maxcap, "X[%i,%i]" % (i, j)))
        X.append(t)
    X_M = [ X[i][j] for i in range(m) for j in range(ng)]

    return  X, X_M, cost, bias

#Objective function

def Obj(X,cost, ng, solver):

    """
    Objective function:

    The Hamiltonian of the system is H= \Sum_{i,j} cost_{i,j} X_{i,j}. Note that in this case H is dimensionless.

    Arguments:
    cost-- (m, ng) cost matrix.
    X-- (m, ng) connection matrix.

    Returns:
    H -- Hamiltonian (cost function)
    objective-- The objective of the problem, i.e. to minimize H subject to constraints
    """
    H = solver.IntVar(0, 1000, "Hamiltonian") #Entries are integers between 0 and 1000 (just a big number)

    solver.Add(H == solver.Sum([ int(cost[i][j]) * X[i][j] for i in range(m)
                                                     for j in range(ng)]) )

    objective = solver.Minimize(H, 1)

    return H, objective

#Enforce Constraints

def constr(X, bias, ng, solver):

    """
    Definition of the Constraints

    Equality constraints correpond to flux conservation in the system. We define the outgoing (positive )flux as the one
    going from source to drain. The Ingoing flux is the reversed one.

    The (iniquality) positive constraint is not needed as it is implemented in the definition of H

    Arguments:
    X -- Connection Matrix to be found.
    bias -- bias vector.
    ng-- dimension of the Children array, eventually containing the ghost.

    """
    #Add equality constraints

    #Outgoing flux (J1)
    [solver.Add(solver.Sum(X[i][j] for j in range(ng)) == bias[i]) for i in range(m)]

    #Ingoing flux (J2)
    [solver.Add( solver.Sum([ X[i][j] for i in range(m)] ) == bias[m+j])for j in range(ng)]

# Count Cutting function as defined at the beginning of the program
def cuts(A, n):

    """
    Counts the number of cuts
    """

    cuts = np.zeros(m)

    for i in range(m):
        l=0
        for j in range(n):
            l+= A[i,j]
        cuts[i]=l-1

    Ncut= sum(cuts)
    return Ncut


#======================================
# Putting all together (Main fucntion)
#======================================

def min_cut(Bars, Children):

    """
    Minimum Cut fuction: assigns chocolate to children while minimizing the minimum number of cuts. It implements the Helper
                         functions defined above.

    Arguments:
    Bars    --- list of chocolate bars and associated available (chocolate) units.
    Children--- list of children and associated demanded (chocolate) units.

    Returns:
    X    --- Matrix having as entries the amount of units moved from Bars to Children
    cuts --- Returns the number of cuts needed to satisfy the demand; this corresponds to the number of connections departing
             from the same "source" node and ending up in different "drain" nodes.

    """
    #np.random.seed(7)
    B = Bars
    C = Children

    global m, n
    m = len(B) # number of bars
    n = len(C) # number of children

    #Initialize the ortools constrained programming solver
    solver = pywrapcp.Solver('Choco')

    # Check satisfiability and implement neutrality
    S , C, ng, f = sat(B, C)

    if f == 0:
        print(S)

    else:
        print(S)

        #initialize variables
        X, X_M, cost, bias = initialize(B,C, solver)

        #Initialize the constraints
        constr(X, bias, ng, solver)

        #Initialize the Objective function
        H , objective= Obj(X,cost, ng, solver)

        #Decision builder: creates the search tree and determines the order in which the solver searches solutions
        db = solver.Phase(X_M,
                  solver.CHOOSE_LOWEST_MIN,
                  solver.ASSIGN_MIN_VALUE)

        #Implement the collector. This variable collects the different solutions found by the optimizer
        collector = solver.LastSolutionCollector()

        # Add decision variables to the collector
        collector.Add(X_M)

        # Add objective function to the collector
        collector.AddObjective(H)

        solver.Solve(db, [objective, collector])

        if collector.SolutionCount() > 0:
            best_solution = collector.SolutionCount() - 1 #Selects the optimal solution
            A= np.zeros([m,n])

            print("Total cost = ", collector.ObjectiveValue(best_solution))
            print()

            for i in range(m):
                for j in range(n):
                    if collector.Value(best_solution, X[i][j]) != 0:
                          A[i,j]= 1
                          print('Bar', i+1, ' assigned', collector.Value(best_solution, X[i][j]) ,'units to child ', j+1,'  Cost = ', cost[i][j])

            print()
            print("Number of cuts=", cuts(A,n))
            print("Time = ", solver.WallTime(), "milliseconds")

        else:  print('No solution has been found')

#=================
#Solution finder
#=================

#------------------------Input data-----------------------------------------------
#Define the inputs as python list objects

Bars= [2,5,7,3] # input bar
Children= [3,2,6,1,2] #input children

#----Obtain Solution

min_cut(Bars, Children)
