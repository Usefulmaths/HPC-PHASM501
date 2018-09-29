# High Performance Computing
This repository contains code that was used for the High Performance Course at UCL (Physics). The course gives a comprehensive overview on how to use high-performance methods for solving different types of partial differential equations (PDEs) over a surface/ volume. 

## Coursework 1
The goal of coursework 1 was to write a solver for diffusion problems of the following form:


<img src="https://latex.codecogs.com/gif.latex?\begin{align}&space;&\nabla\cdot[\sigma(x)\nabla&space;u(x)]&space;=&space;f(x),&space;\text{in}\hspace{0.15cm}\Omega&space;\nonumber&space;\\&space;&u(x)&space;=&space;0,&space;\text{on}&space;\hspace{0.15cm}&space;\partial\Omega\nonumber&space;\end{align}" title="\begin{align} &\nabla\cdot[\sigma(x)\nabla u(x)] = f(x), \text{in}\hspace{0.15cm}\Omega \nonumber \\ &u(x) = 0, \text{on} \hspace{0.15cm} \partial\Omega\nonumber \end{align}" />

Here, Ω is a given domain in two dimensions, f is a given function in Ω, σ is a
scalar field that is defined over Ω and u is the wanted solution.

## Coursework 2
The goal of coursework 2 is to compare different solvers and preconditioners for the diffusion problem

<img src="https://latex.codecogs.com/gif.latex?\begin{align}&space;&\nabla\cdot[\sigma(x)\nabla&space;u(x)]&space;=&space;f(x),&space;\text{in}\hspace{0.15cm}\Omega&space;\nonumber&space;\\&space;&u(x)&space;=&space;0,&space;\text{on}&space;\hspace{0.15cm}&space;\partial\Omega\nonumber&space;\end{align}" title="\begin{align} &\nabla\cdot[\sigma(x)\nabla u(x)] = f(x), \text{in}\hspace{0.15cm}\Omega \nonumber \\ &u(x) = 0, \text{on} \hspace{0.15cm} \partial\Omega\nonumber \end{align}" />

where Ω is a unit square.

## Final Project
The goal of the final project was to implement an overlapping Schwarz decomposition method using MPI parallelisation for the solution of the two-dimensional diffusion problems of the form: 

<img src="https://latex.codecogs.com/gif.latex?\begin{align}&space;&\nabla\cdot[\sigma(x)\nabla&space;u(x)]&space;=&space;f(x),&space;\text{in}\hspace{0.15cm}\Omega&space;\nonumber&space;\\&space;&u(x)&space;=&space;0,&space;\text{on}&space;\hspace{0.15cm}&space;\partial\Omega\nonumber&space;\end{align}" title="\begin{align} &\nabla\cdot[\sigma(x)\nabla u(x)] = f(x), \text{in}\hspace{0.15cm}\Omega \nonumber \\ &u(x) = 0, \text{on} \hspace{0.15cm} \partial\Omega\nonumber \end{align}" />
