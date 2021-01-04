# Implementation of Izhikevich Model
import numpy as np
import matplotlib.pyplot as plt

class izhikevich:
    def __init__(self, I, **kwargs):
        self.a = 0.02
        self.b = 0.2
        self.c = -65
        self.d = 2
        for key, value in kwargs.items():
            if key == 'a':
                self.a = value
            elif key == 'b':
                self.b = value
            elif key == 'c':
                self.c = value
            elif key == 'd':
                self.d = value
        self.threshold = 30  # threshold limit
        self.I = I  # Synaptic currents or injected dc-current
        self.v = -65.0 * np.ones(50000, dtype = float)
        self.u = self.b * self.v  # membrane recovery variable
        self.length = 500  # time in ms
        self.mini_time = 0.01  # 1000/0.50= 2000


def plot_potential_decay(obj):
    arr_time = np.arange(0, obj.length, obj.mini_time)
    for i in range(0, len(arr_time) - 1):
        obj.v[i + 1] = obj.v[i] + obj.mini_time * ((0.04 * (obj.v[i] ** 2)) + (5 * obj.v[i]) + 140 - obj.u[i] + obj.I)
        obj.u[i + 1] = obj.u[i] + obj.mini_time * (obj.a * (obj.b * obj.v[i] - obj.u[i]))
        if obj.v[i + 1] >= obj.threshold:
            obj.v[i + 1] = obj.c
            obj.u[i + 1] = obj.u[i + 1] + obj.d
    plt.plot(arr_time, obj.v)
    plt.xlabel('time (ms)')
    plt.ylabel('Output (mV)')
    plt.show()


def count_spikes(obj):
    num_spikes = 0
    arr_time = np.arange(0, obj.length, obj.mini_time)
    for i in range(0, len(arr_time) - 1):
        obj.v[i + 1] = obj.v[i] + obj.mini_time * ((0.04 * (obj.v[i] ** 2)) + (5 * obj.v[i]) + 140 - obj.u[i] + obj.I)
        obj.u[i + 1] = obj.u[i] + obj.mini_time * (obj.a * (obj.b * obj.v[i] - obj.u[i]))
        if obj.v[i + 1] >= obj.threshold:
            obj.v[i + 1] = obj.c
            obj.u[i + 1] = obj.u[i + 1] + obj.d
            num_spikes += 1
    return num_spikes


def plot_spiking_behavior(obj):
    arr_current = np.arange(1, 30, 0.5)
    num_spikes = []
    for i in arr_current:
        obj.I = i
        num_spikes.append(count_spikes(obj))
    plt.plot(arr_current, num_spikes)
    plt.xlabel('Synaptic current (I)')
    plt.ylabel('Number of spikes')
    plt.show()


a = izhikevich(5)
plot_potential_decay(a)
a = izhikevich(5, c = -50)
plot_potential_decay(a)
#plot_spiking_behavior(a)