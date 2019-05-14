import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve


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

    :return: Q matrix, shape=(segments_num, 2, 2)
    :rtype: np.array
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
    """ Returns LHS matrix (merged Q and E) and RHS matrix used to
     solve the set of equations.

    LHS matrix is the leftmost matrix from the set of the equations.
    It consists of E and Q matrices that are properly positioned.
    The right-hand side of the equation is the RHS matrix.

    :param series: time series in segmented or normal form
    :type series: np.array
    :param segments_num: number of segments in the segmented series
    :type segments_num: int
    :param Q: Q matrix derived from DFA method conditions
              shape=(segments_num, 2, 2)
    :type Q: np.array
    :param E: E matrix derived from continuity condition
              shape=(segments_num-1, 4+(segments_num-2)*2)
    :type E: np.array

    :return: LHS matrix, shape=(matrix_size, matrix_size)
             RHS matrix, shape=matrix_size
    :rtype: (np.array, np.array)
    """

    data, row, col = [], [], []
    segments_size = len(series[0])
    matrix_size = 2*segments_num + (segments_num-1)
    rhs = np.zeros(shape=matrix_size)
    series = series.flatten()
    for w in range(0, segments_num):
        row.extend([w*2] * 2 + [w*2 + 1] * 2)
        col.extend([w*2, w*2+1, w*2, w*2+1])
        data.extend([Q[w][0][0], Q[w][0][1], Q[w][1][0], Q[w][1][1]])
        sum_xi_i = 0
        sum_xi = 0
        for j in range((w*segments_size)+1, (w+1)*segments_size+1):
            sum_xi_i += j * series[j-1]
            sum_xi += series[j-1]
        rhs[2*w] = sum_xi_i
        rhs[2*w + 1] = sum_xi
    for v in range(segments_num*2, matrix_size):
        for i in range(0, 4):
            tmp = E[(v-segments_num*2)][(v-segments_num*2)*2+i]
            row.extend([v, (v-segments_num*2)*2+i])
            col.extend([(v-segments_num*2)*2+i, v])
            data.extend([tmp] * 2)
    lhs = csr_matrix((data, (row, col)), shape=(matrix_size, matrix_size))
    return lhs, rhs


def calculate_trend(segmented_series):
    """ Returns parameters of the calculated continuous linear trend.

    It solves the set of the equations needed to fit a trend line.
    It returns calculated linear function parameters in a form of a list.
    The linear trend is meant to be continuous but it also has to have
    the minimal difference between it and actual time series values.
    The list is returned without the redundant lambda values.

    :param segmented_series: series divided into segments
    :type segmented_series: np.array

    :return: parameters of the trend line, shape=segments_num
    :rtype: np.array
    """

    segments_num = len(segmented_series)
    segments_size = len(segmented_series[0])
    E = continuity_matrix(segments_num, segments_size)
    Q = Q_matrix(segments_num, segments_size)
    lhs, rhs = equation_matrices(segmented_series, segments_num, Q, E)
    parameters = spsolve(lhs, rhs)
    if continuity_test(parameters, segments_size, len(segmented_series)) == 0:
        raise ValueError('Trend line is not continous')
    return parameters[:-segments_num+1]


def continuity_test(parameters, segment_len, segments_num):
    """ Checks whether the fitted line is continuous for every point.

    :param parameters: parameters of the fitted trend lines,
                       shape=segments_num
    :type parameters: np.array
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
