# Optimization-with-Ortools
Using Google's ortools to solve network optimization problems

Recently I have been aske to solve a combinatorial problem for an interview (more on this below). After setting up the model in analytical form I started looking for an optimizer that can handle constranints and matrix objective functions in a way that is as close as possible to the analythincal inplementation. My first choice was [Scipy's contraint optimization](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html) tools, but after spending several hours reading through blog posts to make it works, I finally gave up. One of the problems is that one needs to pass functions to the optimizer as one dimensional optimizer as one dimensional arrays. As this is not too bad for the objective function, it makes handling constraints overcomplicated. Not to mention that you cannot easily pass the optimiser a function without using python's [lambda function](http://www.secnetix.de/olli/Python/lambda_functions.hawk), that again is not alsways the most user friendly.  

On the other hand, Google's ortools provide a powerful environment to solve several optimization problems in a very user friendly way. Althought you can find some informations and tutorials [here](https://developers.google.com/optimization/), the documentation is not always complete. To be fair, there is a nice, althouht incomplete manual [here](http://archive.is/f4wvX). So I decied to post some solutions to nice problems showing details of how this power set of libraries works. 

The minimum cut, chocolate assignement problem

This is a variant of the usual assignment problem: 

There are m chocolate bars of varying (integer) length and n hungry children who want differing amounts of chocolate
(again integer amounts). You can cut the chocolate bars and the goal is to ensure that every child gets the amount they want.
Write a program to distribute the chocolate so that you make the minimum number of cuts.


To solve the problem I mapped it to a minimum cost flow problem, see my blog post for more details. The cost of each "distribution channel" is given by the "distance" between available and required (chocolate) units: cost[i,j]= |B[i]-C[j]|/B[i]. In this way, assignements of resources that do not require a cut have 0 cost, so they always minimize the cost function (e.g. bar: 2 --> child: 2). The rest of the strategy is to make "less costly", configurations that are close to each other.

Note: A cut is the number of times we break up the chocolate bar, so in the given example {2,5,7} --> {3,2,5,1}, the minimum cut is one instead of two as we have

|Bar | Units | Child|
|----|-------|------|
| 1  |   2   |  2   |
| 2  |   5   |  3.  |  
| 3  |   3   |  1.  |
| 3  |   1   |  4.  |

So, bar 3 has been cut into two pieces with a single cut. 





