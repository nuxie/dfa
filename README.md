# DFA
Detrended Fluctuation Analysis with continuous trend line

dfa_simple.py - a simple, basic implementation of the DFA method without the trend line continuity condition; the trend is represented as a quadratic function

continuous_trend.py - given segmented series as an input, it allows to calculate linear trends that are continuous in every "middle point" (uses quadratic programming)

dfa.py - DFA method implementation with the trend line continuity condition; it uses calculate_trend function defined in the continuous_trend.py file; at the moment the trend is linear but it will be implemented as a quadratic function
