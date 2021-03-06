from continous_trend import calculate_trend
from pandas import read_csv
import numpy as np
import matplotlib.pyplot as plt


def random_walk(series, length):
    """ Converts the series into a random walk using cumulative sum.

    :param series: time series to be converted
    :type series: numpy.array

    :param length: length of the resulting random walk
    :type length: int
    """

    mean = np.mean(series)
    previous = series[0] - mean
    for i in range(0, length):
        current = previous + series[i] - mean
        yield current
        previous = current


def fluctuation(segmented_series, parameters):
    """ Returns the fluctuation value for the segmented series.

    Fluctuation - root-mean-square deviation from the fitted trend.

    :param segmented_series: series divided into segments
    :type segmented_series: numpy.array
    :param parameters: parameters of the fitted trend lines,
                       shape=len(segmented_series)=number of segments
    :type parameters: numpy.array

    :return: fluctuation value for the segmented series
    :rtype: float
    """

    local_trend = np.zeros(shape=np.shape(segmented_series))
    segment_size = len(segmented_series[0])
    for i in range(len(segmented_series)):
        x_values = np.arange(i*segment_size+1, (i+1)*segment_size+1)
        local_trend[i] = np.polyval([parameters[2*i], parameters[2*i + 1]], x_values)
    rms = np.zeros(np.shape(segmented_series)[0])
    num = 0
    for actual, estimated in zip(segmented_series, local_trend):
        rms[num] = np.sqrt(np.mean((actual-estimated)**2))
        num += 1
    return np.mean(rms)


def exponent(F, n, plot):
    """ Fits a line into log-log graph of fluctuations against segments
        sizes and returns its parameters - slope and intercept.

    :param F: fluctuations calculated for segmented series of each size
    :type F: numpy.array
    :param n: segments sizes corresponding to F values
    :type n: numpy.array
    :param plot: indicates if the plot of the fitted line should be shown
    :type plot: int

    :return: parameters of the line fitted into the log-log graph
    :rtype: list
    """

    A = np.vstack([np.log(n), np.ones(len(n))]).T
    res = np.linalg.lstsq(A, np.log(F), rcond=None)[0]
    if plot:
        plt.plot(np.log(n), np.log(F), 'o')
        plt.plot(np.log(n), res[0]*np.log(n) + res[1])
        plt.xlabel("log(n)")
        plt.ylabel("log(F(n))")
        plt.show()
    return res


def dfa(series, plot=0):
    """ Returns the slope of the line fitted into log-log graph of
        fluctuations against segments sizes, which is the DFA exponent.

    :param series: time series for which to calculate the DFA exponent
    :type series: numpy.array
    :param plot: optional, defaults to 0
                 indicates if the plot of the fitted line should be shown
    :type plot: int

    :return: slope of the line fitted into log-log graph
    :rtype: float
    """

    data_length = len(series)
    data = list(random_walk(series, data_length))
    max_exponent = np.log2(data_length)
    segments_sizes = (2 ** np.arange(2, max_exponent)).astype(int)
    fluctuations = np.zeros(len(segments_sizes))
    for i, size in enumerate(segments_sizes):
        segmented_series = np.reshape(data, ((data_length//size), size))
        parameters = calculate_trend(segmented_series)
        fluctuations[i] = fluctuation(segmented_series, parameters)
    return exponent(fluctuations, segments_sizes, plot)[0]


url = 'https://raw.githubusercontent.com/nuxie/dfa/master/time_series.txt'
ts = read_csv(url, header=None)
ts = np.array(ts.values.flatten())

# white noise - DFA exponent should be near 0.5:
# ts = np.random.standard_normal(size=8192)

print("DFA exponent for given time series with continous trend lines equals:", dfa(ts, plot=1))
