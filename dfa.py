from continous_trend import calculate_trend
from pandas import read_csv
import numpy as np
import matplotlib.pyplot as plt


def random_walk(series, length):
    mean = np.mean(series)
    previous = series[0] - mean
    for i in range(0, length):
        current = previous + series[i] - mean
        yield current
        previous = current


def fluctuation(segmented_series, parameters):
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
    # fits a line into log-log graph of fluctuations against segments sizes and returns its slope
    # plots a graph if requested
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

