import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":

    f = 10      # [Hz]
    tau = 1     # [s]

    Ts = np.min([1/f, tau])/10     # Nyquist theorem is /2

    tspan = [0, 10]

    t = np.arange(tspan[0], tspan[1], Ts)

    A = 1
    pulse = A*np.sin(2*np.pi*f*t[t<tau])

    # Signal
    signal = np.zeros(t.shape)
    signal[t<tau] = pulse

    # FFT
    npoints = signal.size*10
    S = np.fft.rfft(signal, npoints)
    freq = np.fft.rfftfreq(npoints, Ts)
    Sabs = np.abs(S)*Ts
    Sphase = np.angle(S)

    # Echoes
    t1 = 3
    k1 = 0.5
    t2 = 5
    k2 = 0.3
    echo = np.zeros(t.shape)
    echo[(t>=t1) & (t<t1+tau)] = k1*pulse[::-1]
    echo[(t>=t2) & (t<t2+tau)] = k2*pulse[::-1]

    # Matched filter
    matched = np.convolve(signal, echo)*Ts

    # Indistinguishable echoes
    t1 = 3
    k1 = 0.5
    t2 = 4
    k2 = 0.3
    echo2 = np.zeros(t.shape)
    echo2[(t>=t1) & (t<t1+tau)] = k1*pulse[::-1]
    echo2[(t>=t2) & (t<t2+tau)] = k2*pulse[::-1]

    # Matched filter
    matched2 = np.convolve(signal, echo2)*Ts
 
    # Plot of Signal
    # Plot of FFT
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(t, signal)
    plt.title("Simple Pulse")
    plt.xlabel("time [s]")
    plt.ylabel("amplitude")
    plt.grid(True)
    plt.subplot(3, 1, 2)
    plt.plot(freq, Sabs)
    plt.title("Fourier Transform of the Simple Pulse")
    plt.ylabel("amplitude")
    plt.grid(True)
    plt.subplot(3, 1, 3)
    plt.plot(freq, Sphase)
    plt.xlabel("frequency [Hz]")
    plt.ylabel("phase [rad/s]")
    plt.grid(True)
    plt.tight_layout()

    # Plot of signal and echoes
    # Plot of matched filter response
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(t, signal, t, echo)
    plt.title("Simple Pulse and 2 Echoes")
    plt.xlabel("time [s]")
    plt.ylabel("amplitude")
    plt.legend(["signal", "echoes"])
    plt.grid(True)
    plt.subplot(2, 1, 2)
    plt.plot(t, matched[:t.size])
    plt.title("Matched filter")
    plt.xlabel("time [s]")
    plt.ylabel("amplitude")
    plt.grid(True)
    plt.tight_layout()

    # Plot of signal and indistinguishable echoes
    # Plot of matched filter response
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(t, signal, t, echo2)
    plt.title("Simple Pulse and 2 Indistinguishable Echoes")
    plt.xlabel("time [s]")
    plt.ylabel("amplitude")
    plt.legend(["signal", "echoes"])
    plt.grid(True)
    plt.subplot(2, 1, 2)
    plt.plot(t, matched2[:t.size])
    plt.title("Matched filter")
    plt.xlabel("time [s]")
    plt.ylabel("amplitude")
    plt.grid(True)
    plt.tight_layout()

    # Show plots
    plt.show()
