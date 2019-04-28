# DFA

Detrended Fluctuation Analysis with continuous trend line.
Detailed information on the algorithm can be found on Wikipedia (https://en.wikipedia.org/wiki/Detrended_fluctuation_analysis).

This repository consists of 3 files:

* dfa_simple.py - a simple, basic implementation of the DFA method without the trend line continuity condition; the trend is represented as a quadratic function

* continuous_trend.py - given segmented series as an input, it allows to calculate linear trends that are continuous in every "middle point" (uses quadratic programming)

* dfa.py - DFA method implementation with the trend line continuity condition; it uses calculate_trend function defined in the continuous_trend.py file; at the moment the trend is linear but it will be implemented as a quadratic function

<br>

<b>Mathematics behind finding the DFA exponent for the continuous trend line</b>

<br>

We are looking for the minimum of the given form:


![form](http://latex.codecogs.com/png.latex?%5Cdpi%7B150%7D%20F%20%3D%20%5Cfrac%7B1%7D%7B2%7Dp%5ETQp%20&plus;%20c%5ETp) where 

with a continuity condition given by the following equation:

![condition](http://latex.codecogs.com/png.latex?%5Cdpi%7B150%7D%20Ep%20%3D%20d)

It comes down to solving the following set of equations:

![equations](http://latex.codecogs.com/png.latex?%5Cdpi%7B150%7D%20%5Cbegin%7Bbmatrix%7D%20Q%20%26%20E%5ET%20%5C%5C%20E%20%26%200%20%5Cend%7Bbmatrix%7D%20%5Cbegin%7Bbmatrix%7D%20p%20%5C%5C%20%5Clambda%20%5Cend%7Bbmatrix%7D%20%3D%20%5Cbegin%7Bbmatrix%7D%20-c%20%5C%5C%20d%20%5Cend%7Bbmatrix%7D)
