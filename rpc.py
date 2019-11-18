import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    f = 10      # [Hz]
    tau = 1     # [s]

    Ts = np.min([1/f, tau])/10     # Nyquist theorem is /2

    tspan = [0, 7]

    t = np.arange(tspan[0], tspan[1], Ts)

    A = 1
    s = np.empty(t.shape)

    for i, ti in enumerate(t):
        if ti>0 and ti<tau:
            s[i] = A*np.cos(2*np.pi*f*ti)
        else:
            s[i] = 0

    plt.figure()
    plt.plot(t, s)
    plt.xlabel("time [s]")
    plt.ylabel("amplitude")
    plt.title("Simple Pulse")








    plt.show()    
