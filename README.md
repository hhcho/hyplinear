# hyplinear

This repository contains a MATLAB implementation of linear support vector machines (SVM) for data points in hyperbolic space. The manuscript describing this method is currently under review. Preprint is available here: [https://arxiv.org/abs/1806.00437](https://arxiv.org/abs/1806.00437).

##### Dependencies
- MATLAB (version R2017b)
- [LIBLINEAR](https://www.csie.ntu.edu.tw/~cjlin/liblinear/) for Euclidean support vector classification. Also used as a subroutine in our hyperbolic SVM implementation. Developed with version 2.20.

##### How to Run
We provide example scripts for comparing hyperbolic SVM to Euclidean SVM on three benchmark datasets described in our manuscript:
* Gaussian point clouds in hyperbolic space (`run_gaussian.m`)
* simulated scale-free networks with simulated node labels, embedded in hyperbolic space via LaBNE \[1\] (`run_randnet.m`)
* real-world networks with known node labels, embedded in hyperbolic space using the approach of Chamberlain et al. \[2\] (`run_realnet.m`).

Before executing these scripts, edit the following top line in each script
```matlab
addpath /PATH/TO/LIBLINEAR/matlab % modify
```
to include the correct path to the MATLAB subdirectory of LIBLINEAR.

##### Contact
Hoon Cho, hhcho@mit.edu

##### References

\[1\] G. Alanis-Lobato, P. Mier, and M.A. Andrade-Navarro. Efficient Embedding of Complex Networks to Hyperbolic Space via their Laplacian. Scientific reports, 6:30108, 2016.

\[2\] B.P. Chamberlain, J. Clough, and M.P. Deisenroth. Neural Embeddings of Graphs in Hyperbolic Space. arXiv preprint, arXiv:1705.10359, 2017.

