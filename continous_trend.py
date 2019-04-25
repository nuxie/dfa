import numpy as np


def continuity_matrix(segments_num, segments_size):
    # returns E matrix responsible for continuity used on lhs of the equation
    # continuity condition: Ep = d checked in (segments_num-1) "middle points"
    # number of rows - number of "middle points" where continuity is checked
    # number of columns - 4 for one row, each additional row adds 2 columns
    E = np.zeros(shape=(segments_num-1, 4+(segments_num-2)*2))
    for i in range(0, segments_num-1):
        tmp = (i+1)*segments_size + 0.5
        E[i][i*2] = tmp
        E[i][i*2 + 1] = 1
        E[i][i*2 + 2] = -tmp
        E[i][i*2 + 3] = -1
    return E


def Q_matrix(segments_num, segments_size):
    # returns matrix partially responsible for DFA method conditions
    # matrix Q - used on lhs of the equation, trend line values
    Q = np.zeros(shape=(segments_num, 2, 2))
    for i in range(0, segments_num):
        sum_i_sq = 0
        sum_i = 0
        for j in range((i*segments_size)+1, (i+1)*segments_size+1):
            sum_i_sq += j**2
            sum_i += j
        Q[i][0][0] = sum_i_sq
        Q[i][0][1] = sum_i
        Q[i][1][0] = sum_i
        Q[i][1][1] = segments_size
    return Q


def equation_matrices(series, segments_num, Q, E):
    # matrix M - used on lhs of the equation, merged Q and E matrices
    # matrix cd - used on rhs of the equation, time series values
    segments_size = len(series[0])
    matrix_size = 2*segments_num + (segments_num-1)
    cd = np.zeros(shape=matrix_size)
    M = np.zeros(shape=(matrix_size, matrix_size))
    series = series.flatten()
    for w in range(0, segments_num):
        M[w*2][w*2] = Q[w][0][0]
        M[w*2][w*2 + 1] = Q[w][0][1]
        M[w*2 + 1][w*2] = Q[w][1][0]
        M[w*2 + 1][w*2 + 1] = Q[w][1][1]
        sum_xi_i = 0
        sum_xi = 0
        for j in range((w*segments_size)+1, (w+1)*segments_size+1):
            sum_xi_i += j * series[j-1]
            sum_xi += series[j-1]
        cd[2*w] = sum_xi_i
        cd[2*w + 1] = sum_xi
    for v in range(segments_num*2, matrix_size):
        for i in range(0, 4):
            tmp = E[(v-segments_num*2)][(v-segments_num*2)*2+i]
            M[v][(v-segments_num*2)*2+i] = tmp
            M[(v-segments_num*2)*2+i][v] = tmp
    return M, cd


def calculate_trend(segmented_series):
    segments_num = len(segmented_series)
    segments_size = len(segmented_series[0])
    E = continuity_matrix(segments_num, segments_size)
    Q = Q_matrix(segments_num, segments_size)
    lhs, rhs = equation_matrices(segmented_series, segments_num, Q, E)
    parameters = np.linalg.solve(lhs, rhs) # LAPACK routine _gesv - uses LU decomposition
    if continuity_test(parameters, segments_size, len(segmented_series)) == 0:
        raise ValueError('Trend line is not continous')
    return parameters[:-segments_num+1]


def continuity_test(parameters, segment_len, segments_num):
    testing_points = np.zeros(segments_num-1)
    for i in range(0, segments_num-1):
        testing_points[i] = (i+1)*segment_len + 0.5
        val1 = testing_points[i]*parameters[2*i] + parameters[2*i+1]
        val2 = testing_points[i]*parameters[2*i+2] + parameters[2*i+3]
        if round(val1, 4) != round(val2, 4):
            return 0
    return 1
