# Implementation of Hodgkin-Huxley Model
import numpy as np
import math
import matplotlib.pyplot as plt


class hh:
    def __init__(self, I, **kwargs):
        self.init_potential = -65  # initial potential

        self.V_Na = 115.0  # sodium potential
        self.V_K = -12.0  # potassium potential
        self.V_L = 10.6  # leak potential
        self.I = I
        self.length = 100  # time in ms
        self.mini_time = 0.01  # 1000/0.50= 2000
        self.V = np.zeros(10000, dtype=float)
        self.n = np.zeros(10000, dtype=float)
        self.m = np.zeros(10000, dtype=float)
        self.h = np.zeros(10000, dtype=float)
        self.g_Na = 120.0
        self.g_K = 36.0
        self.g_L = 0.3
        self.C = 1.0  # Membrane capacitance


def plot_potential_decay(obj):
    arr_time = np.arange(0, obj.length, obj.mini_time)
    # obj.V[0] = 0 #base line voltage
    obj.n[0] = alpha_n(obj.V[0]) / (alpha_n(obj.V[0]) + beta_n(obj.V[0]))
    obj.m[0] = alpha_m(obj.V[0]) / (alpha_m(obj.V[0]) + beta_m(obj.V[0]))
    obj.h[0] = alpha_h(obj.V[0]) / (alpha_h(obj.V[0]) + beta_h(obj.V[0]))

    for i in range(0, len(arr_time) - 1):
        a_n = alpha_n(obj.V[i])
        b_n = beta_n(obj.V[i])
        a_m = alpha_m(obj.V[i])
        b_m = beta_m(obj.V[i])
        a_h = alpha_h(obj.V[i])
        b_h = beta_h(obj.V[i])

        I_na = (obj.m[i] ** 3) * obj.g_Na * obj.h[i] * (obj.V[i] - obj.V_Na)
        I_k = obj.g_K * (obj.n[i] ** 4) * (obj.V[i] - obj.V_K)
        I_l = obj.g_L * (obj.V[i] - obj.V_L)
        I_total = obj.I[i] - I_k - I_na - I_l

        obj.V[i + 1] = obj.V[i] + obj.mini_time * (I_total / obj.C)
        # if obj.V[i + 1] <= 0:
        # print(obj.V[i + 1])
        obj.n[i + 1] = obj.n[i] + obj.mini_time * (a_n * (1 - obj.n[i]) - (b_n * obj.n[i]))
        obj.m[i + 1] = obj.m[i] + obj.mini_time * (a_m * (1 - obj.m[i]) - (b_m * obj.m[i]))
        obj.h[i + 1] = obj.h[i] + obj.mini_time * (a_h * (1 - obj.h[i]) - (b_h * obj.h[i]))

    obj.V[:] += obj.init_potential
    plt.plot(arr_time, obj.V)
    plt.xlabel('time (ms)')
    plt.ylabel('Output (mV)')
    plt.show()


def alpha_m(V):
    return 0.1 * ((25.0 - V) / (math.exp((25.0 - V) / 10.0) - 1.0))


def beta_m(V):
    return 4 * math.exp(-V / 18)


def alpha_h(V):
    return 0.07 * math.exp(-V / 20)


def beta_h(V):
    return 1.0 / (math.exp((30 - V) / 10) + 1)


def alpha_n(V):
    return 0.01 * ((10.0 - V) / (math.exp((10 - V) / 10.0) - 1.0))


def beta_n(V):
    return 0.125 * math.exp(-V / 80.0)


arr = np.zeros(10000, dtype=float)
arr[2500 : 3500] = 5
arr[8500 : 9000] = 50
a = hh(arr)
plot_potential_decay(a)
arr[:] = 50
a = hh(arr)
plot_potential_decay(a)
# print(sum(a.m) , sum(a.n), sum(a.h), sum(a.V))
