import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import chirp, hilbert, welch


def fft(t, s, npoints_factor=1):
    npoints = int(s.size*npoints_factor)

    Ts = np.mean(np.diff(t))
    freq = np.fft.rfftfreq(npoints, Ts)

    S = np.fft.rfft(s, npoints)
    Sabs = np.abs(S)*Ts
    Sabs[1:] *= 2
    Sphase = np.angle(S)

    return freq, Sabs, Sphase

def envelope(signal):
    analytic = hilbert(signal)
    envelope = np.abs(analytic)
    return envelope


if __name__ == "__main__":
    f = 10      # [Hz]
    tau = 1     # [s]
    Ts = np.min([1/f, tau])/20     # Nyquist theorem is /2

    tspan = [0, 10]
    t = np.arange(tspan[0], tspan[1], Ts)

    # Pulse
    A = 1
    tpulse = t[t<tau]
    pulse = np.sin(2*np.pi*f*tpulse)
    code = "+++--+-"
    l = len(code)
    n = 2
    n = np.max([n, 1])
    tphase = n/2/f
    phase = np.ones(pulse.shape)
    
    for i in range(int(tau/tphase+0.5)):
        if code[i%l] == '-':
            #pulse[(tpulse>=i*tphase) & (tpulse<(i+1)*tphase)] *= -1
            phase[(tpulse>=i*tphase) & (tpulse<(i+1)*tphase)] = -1

    pulse = pulse*phase

    # Signal
    signal = np.zeros(t.shape)
    signal[t<tau] = pulse

    # Signal envelope (simple pulse)
    envelope_s = envelope(signal)

    # FFT
    freq, Sabs, Sphase = fft(t, pulse, 10)

    # Matched filter response and FFT
    tm = t[:pulse.size]
    matched = pulse[::-1]
    Mfreq, Mabs, Mphase = fft(tm, matched, 10)

    # Echo 1
    t1 = 3
    k1 = 0.5
    echo1 = np.zeros(t.shape)
    echo1[(t>=t1) & (t<t1+tau)] = k1*pulse
    
    # Echo 2
    t2 = 6
    k2 = 0.3
    echo2 = np.zeros(t.shape)
    echo2[(t>=t2) & (t<t2+tau)] = k2*pulse

    # Both echoes
    echoes = echo1 + echo2

    # Matched filter output both echoes
    output = np.convolve(matched, echoes)*Ts
    output = output[:t.size]

    # Envelope of matched filter output
    envelope_o = envelope(output)

    # Echo 3 (indistinguishable)
    t3 = 4
    k3 = 0.3
    echo3 = np.zeros(t.shape)
    echo3[(t>=t3) & (t<t3+tau)] = k3*pulse

    # Both echoes (indistinguishable)
    echoesi = echo1 + echo3
    
    # Matched filter output both echoes (indistinguishable)
    outputi = np.convolve(matched, echoesi)*Ts
    outputi = outputi[:t.size]

    # Envelope of matched filter output (indistinguishable)
    envelope_oi = envelope(outputi)

    # Noisy echoes
    N = 0.5
    noise = np.random.normal(0, N, t.size)
    noisy_echoes = echoes + noise

    # Matched filter output both echoes with noise
    noisy_output = np.convolve(matched, noisy_echoes)*Ts
    noisy_output = noisy_output[:t.size]

    # Envelope of matched filter output with noise
    noisy_envelope_o = envelope(noisy_output)

    # Noise FFT
    Nfreq, Nabs = welch(noise, 1/Ts, nperseg=noise.size)

    # Plot of Signal
    # Plot of FFT
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(tpulse, pulse)
    plt.plot(tpulse[tpulse<tphase*l], phase[tpulse<tphase*l], 'r', lw=1)
    plt.title("Binary Pulses")
    plt.xlabel("time [s]")
    plt.ylabel("amplitude")
    plt.legend(["signal", "baker code"], loc='upper right')
    plt.grid(True)
    plt.subplot(3, 1, 2)
    plt.plot(freq, Sabs)
    plt.title("Fourier Transform of the Binary Pulses")
    plt.ylabel("amplitude")
    plt.grid(True)
    plt.subplot(3, 1, 3)
    plt.plot(freq, Sphase)
    plt.xlabel("frequency [Hz]")
    plt.ylabel("phase [rad/s]")
    plt.grid(True)
    plt.tight_layout()

    # Plot of Matched Filter
    # Plot of Matched Filter FFT
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(tm, matched)
    plt.title("Matched filter time response")
    plt.xlabel("time [s]")
    plt.ylabel("amplitude")
    plt.grid(True)
    plt.subplot(3, 1, 2)
    plt.plot(Mfreq, Mabs)
    plt.title("Matched filter frequency response")
    plt.ylabel("amplitude")
    plt.grid(True)
    plt.subplot(3, 1, 3)
    plt.plot(Mfreq, Mphase)
    plt.xlabel("frequency [Hz]")
    plt.ylabel("phase [rad/s]")
    plt.grid(True)
    plt.tight_layout()

    # Plot of signal and echoes
    # Plot of matched filter response
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(t, signal, t, echoes)
    plt.title("Binary Pulses and 2 Echoes")
    plt.xlabel("time [s]")
    plt.ylabel("amplitude")
    plt.legend(["signal", "echoes"])
    plt.grid(True)
    plt.subplot(2, 1, 2)
    plt.plot(t, output, t, envelope_o, 'k', lw=0.5)
    plt.title("Matched filter output")
    plt.xlabel("time [s]")
    plt.ylabel("amplitude")
    plt.legend(["output", "envelope"])
    plt.grid(True)
    plt.tight_layout()

    # Plot of signal and indistinguishable echoes
    # Plot of matched filter response
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(t, signal, t, echoesi)
    plt.title("Binary Pulses and 2 Indistinguishable Echoes")
    plt.xlabel("time [s]")
    plt.ylabel("amplitude")
    plt.legend(["signal", "echoes"])
    plt.grid(True)
    plt.subplot(2, 1, 2)
    plt.plot(t, outputi, t, envelope_oi, 'k', lw=0.5)
    plt.title("Matched filter output")
    plt.xlabel("time [s]")
    plt.ylabel("amplitude")
    plt.legend(["output", "envelope"])
    plt.grid(True)
    plt.tight_layout()

    # Plot of signal and echoes with noise
    # Plot of matched filter response with noise
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(t, signal, t, noisy_echoes)
    plt.title("Binary Pulses and 2 Echoes with Noise")
    plt.xlabel("time [s]")
    plt.ylabel("amplitude")
    plt.legend(["signal", "echoes"])
    plt.grid(True)
    plt.subplot(2, 1, 2)
    plt.plot(t, noisy_output, t, noisy_envelope_o, 'k', lw=0.5)
    plt.title("Matched filter output")
    plt.xlabel("time [s]")
    plt.ylabel("amplitude")
    plt.legend(["output", "envelope"])
    plt.grid(True)
    plt.tight_layout()

    # Plot of Noise
    # Plot of FFT
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(t, noise)
    plt.title("Received Noise")
    plt.xlabel("time [s]")
    plt.ylabel("amplitude")
    plt.grid(True)
    plt.subplot(2, 1, 2)
    plt.plot(Nfreq, Nabs)
    plt.title("Noise PSD")
    plt.ylabel("power")
    plt.grid(True)
    plt.tight_layout()


    # Show plots
    plt.show()
