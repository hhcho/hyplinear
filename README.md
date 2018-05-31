# hyplinear

This repository contains a MATLAB implementation of linear support vector classification for data points in hyperbolic space. The manuscript describing this method is currently under review.

##### Dependencies
- MATLAB (version R2017b)
- [LIBLINEAR](https://www.csie.ntu.edu.tw/~cjlin/liblinear/) for Euclidean support vector classification. Also used as a subroutine in our hyperbolic SVM implementation. Developed with version 2.20.

##### How to Run
We provide example simulated datasets in `data/gaussian`, each of which consists of 400 points sampled from a four-component Gaussian mixture model defined in the 2D hyperbolic space. The coordinates are given in the Poincare disk model with unit radius.

We also provide a script `run_gaussian.m` that iterates through the example datasets and compares the classification performance of hyperbolic vs Euclidean linear-SVMs via cross-validation.

##### Contact
Hoon Cho, hhcho@mit.edu
