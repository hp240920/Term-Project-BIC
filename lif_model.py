import numpy as np
import matplotlib.pyplot as plt


class lif:
    def __init__(self, I= 0, **kwargs):
        # define variables
        self.const_cap = 10
        self.voltage = np.zeros(100, dtype=float)
        self.resistance = 1
        self.threshold = 0.1
        self.time_cost = self.const_cap * self.resistance
        self.v_spike = 0.5
        self.ref_time = 4  # refractory time ms after spike
        self.reset_v = 0
        self.I = I
        self.length = 100  # time in ms
        self.mini_time = 1  # ??? 1000/0.50= 2000


def plot_potential_decay(obj):
    isSpike = False
    time = 0
    arr_time = np.arange(0, obj.length, obj.mini_time)
    for i in range(0, len(arr_time) - 1):

        if isSpike:
            obj.voltage[i + 1] = obj.reset_v
            isSpike = False
            continue
        if arr_time[i] < time:
            continue

        obj.voltage[i + 1] = obj.voltage[i] + obj.mini_time * (
                    (obj.I - (obj.voltage[i] / obj.resistance)) / obj.const_cap)

        if obj.voltage[i + 1] >= obj.threshold:
            obj.voltage[i + 1] += obj.v_spike
            time = arr_time[i] + obj.ref_time
            # print(arr_time[i])
            isSpike = True

    plt.plot(arr_time, obj.voltage)
    plt.xlabel('time (ms)')
    plt.ylabel('Output (mV)')
    plt.show()


def count_spikes(obj):
    spike_count = 0
    isSpike = False
    time = 0
    arr_time = np.arange(0, obj.length, obj.mini_time)
    for i in range(0, len(arr_time) - 1):

        if isSpike:
            obj.voltage[i + 1] = obj.reset_v
            isSpike = False
            continue
        if arr_time[i] < time:
            continue

        obj.voltage[i + 1] = obj.voltage[i] + obj.mini_time * (
                    (obj.I - (obj.voltage[i] / obj.resistance)) / obj.const_cap)

        if obj.voltage[i + 1] >= obj.threshold:
            obj.voltage[i + 1] += obj.v_spike
            time = arr_time[i] + obj.ref_time
            spike_count += 1
            isSpike = True
    # print(spike_count)
    return spike_count


def plot_spiking_behavior():
    arr_current = np.arange(1, 5, 0.05)
    num_spikes = []
    for i in arr_current:
        obj = lif(i)
        num_spikes.append(count_spikes(obj))
    # print(num_spikes)
    plt.plot(arr_current, num_spikes)
    plt.xlabel('Synaptic current (I)')
    plt.ylabel('Number of spikes')
    plt.show()


if __name__ == '__main__':
    a = lif(0.9)
    print(count_spikes(a))
    plot_potential_decay(a)
    # plot_spiking_behavior()
