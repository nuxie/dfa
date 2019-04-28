import numpy as np


def continuity_matrix(segments_num, segments_size):
    """ Returns E matrix derived from continuity condition.

    E matrix is used on the left-hand side of the set of equations.
    It is represented as a two dimensional list.
    Number of rows equals number of segments - 1.
    For each row there are 4 non-zero elements.

    :param segments_num: number of segments in the segmented series
    :type segments_num: int
    :param segments_size: size of one segment in the segmented series
    :type segments_size: int

    :return: E matrix in a form of list
    :rtype: 2dim list, shape=(segments_num-1, 4+(segments_num-2)*2)
    """

    E = np.zeros(shape=(segments_num-1, 4+(segments_num-2)*2))
    for i in range(0, segments_num-1):
        tmp = (i+1)*segments_size + 0.5
        E[i][i*2] = tmp
        E[i][i*2 + 1] = 1
        E[i][i*2 + 2] = -tmp
        E[i][i*2 + 3] = -1
    return E


def Q_matrix(segments_num, segments_size):
    """ Returns Q matrix derived from DFA method conditions.

    Q matrix is used on the left-hand side of the set of equations.
    It is derived from the difference between trend line and actual
    time series values that we are trying to minimise.
    It is represented as a three dimensional list.
    Number of "Q's" equals number of segments.
    Each "Q" is a list of 2x2 dimensions.

    :param segments_num: number of segments in the segmented series
    :type segments_num: int
    :param segments_size: size of one segment in the segmented series
    :type segments_size: int

    :return: Q matrix in a form of a list
    :rtype: 3dim list, shape=(segments_num, 2, 2)
    """

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
    """ Returns M matrix (merged Q and E) and cd matrix used on the
    right-hand side of the set of equations.

    M matrix is the leftmost matrix from the set of the equations.
    It consists of E and Q matrices that are properly positioned.
    The right-hand side of the equation is the cd matrix.

    :param series: time series in segmented or normal form
    :type series: list
    :param segments_num: number of segments in the segmented series
    :type segments_num: int
    :param Q: Q matrix derived from DFA method conditions
    :type Q: 3dim list, shape=(segments_num, 2, 2)
    :param E: E matrix derived from continuity condition
    :type E: 2dim list, shape=(segments_num-1, 4+(segments_num-2)*2)

    :return: M and cd, two matrices in the form of lists needed to solve
            the set of equations
    :rtype: 2dim list shape=(matrix_size, matrix_size),
            1dim list shape=matrix_size
    """

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
    """ Returns parameters of the calculated continuous linear trend.

    It solves the set of the equations needed to fit a trend line.
    It returns calculated linear function parameters in a form of a list.
    The linear trend is meant to be continuous but it also has to have
    the minimal difference between it and actual time series values.
    The list is returned without the redundant lambda values.

    :param segmented_series: series divided into segments
    :type segmented_series: 2dim list

    :return: parameters of the trend line
    :rtype: list, shape=segments_num
    """

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
    """ Checks whether the fitted line is continuous for every point.

    :param parameters: parameters of the fitted trend lines
    :type parameters: list, shape=segments_num
    :param segment_len: size of one segment in the segmented series
    :type segment_len: int
    :param segments_num: number of segments in the segmented series
    :type segments_num: int

    :return: value indicating whether continuity condition is satisfied
    :rtype: boolean
    """

    testing_points = np.zeros(segments_num-1)
    for i in range(0, segments_num-1):
        testing_points[i] = (i+1)*segment_len + 0.5
        val1 = testing_points[i]*parameters[2*i] + parameters[2*i+1]
        val2 = testing_points[i]*parameters[2*i+2] + parameters[2*i+3]
        if round(val1, 4) != round(val2, 4):
            return 0
    return 1
