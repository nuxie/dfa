import pandas
import numpy as np
import matplotlib.pyplot as plt


def random_walk(series, length):
    # converts the series into a "random walk" - cumulative sum
    mean = np.mean(series)
    previous = series[0] - mean
    for i in range(0, length):
        current = previous + series[i] - mean
        yield current
        previous = current


def fluctuation(segmented_series):
    # calculates root mean square deviation for each segment and returns its average
    # also checks the continuity of estimated trend line
    continuity = 1
    local_trend = np.zeros(shape=np.shape(segmented_series))
    parameters = np.zeros(shape=(len(segmented_series), 3))
    segment_size = len(segmented_series[0])
    for n, segment in enumerate(segmented_series):
        x = np.arange(n*segment_size+1, (n+1)*segment_size+1)
        parameters[n] = np.polyfit(x, segment, 2)
    for i in range(len(parameters)):
        x_values = np.arange(i*segment_size+1, (i+1)*segment_size+1)
        local_trend[i] = np.polyval(parameters[i], x_values)
    rms = np.zeros(np.shape(segmented_series)[0])
    num = 0
    for actual, estimated in zip(segmented_series, local_trend):
        rms[num] = np.sqrt(np.mean((actual-estimated)**2))
        num += 1
    return np.mean(rms), continuity, parameters


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
    continuity = np.zeros(len(segments_sizes))
    for i, size in enumerate(segments_sizes):
        segmented_series = np.reshape(data, ((data_length//size), size))
        fluctuations[i], continuity[i], parameters = fluctuation(segmented_series)
    return exponent(fluctuations, segments_sizes, plot)[0]


url = 'https://raw.githubusercontent.com/nuxie/dfa/master/time_series.txt'
ts = pandas.read_csv(url, header=None)
ts = np.array(ts.values.flatten())

# white noise - DFA exponent should be near 0.5:
# ts = np.random.standard_normal(size=2048)

print("DFA exponent for given time series equals", dfa(ts, plot=1), "(quadratic trends).")