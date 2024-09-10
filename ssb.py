#IMPLEMENTATION OF SSB AMPLITUDE MODULATION IN PYTHON

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.io import wavfile
import sounddevice as sd

# Function to compute the Hilbert transform for SSB modulation
def hilbert_transform(x, N):
    n = np.arange(N)  # Create an array of indices
    h = np.zeros(N)   # Initialize Hilbert kernel with zeros
    h[0] = 1          # DC component
    h[1::2] = 2 / (np.pi * n[1::2])  # Hilbert transform kernel components
    return signal.convolve(x, h, mode='same')  # Convolve input signal with Hilbert kernel

# Function to perform Single Sideband (SSB) modulation
def ssb_modulate(x, fc, fs, N, lower_sideband=False):
    analytic_signal = x + 1j * hilbert_transform(x, N)  # Create analytic signal
    t = np.arange(len(x)) / fs  # Time vector
    carrier = np.exp(1j * 2 * np.pi * fc * t)  # Carrier signal
    # Modulate the signal and return either upper or lower sideband
    return np.real(analytic_signal * (np.conj(carrier) if lower_sideband else carrier))

# Function to perform SSB demodulation
def ssb_demodulate(y, fc, fs, lower_sideband=False):
    t = np.arange(len(y)) / fs  # Time vector
    carrier = np.exp(-1j * 2 * np.pi * fc * t)  # Carrier signal for demodulation
    if lower_sideband:
        carrier = np.conj(carrier)  # Use conjugate for lower sideband
    demodulated = y * carrier  # Multiply signal by carrier
    nyq = 0.5 * fs  # Nyquist frequency
    # Low-pass filter to extract baseband signal
    b, a = signal.butter(6, 0.1 * nyq / nyq, btype='low')
    return np.real(signal.filtfilt(b, a, demodulated))

# Animation parameters
chunk_size = 1024  # Size of chunks to process for animation
interval = (chunk_size / 44100) * 1000  # Time interval for animation frames in milliseconds

# Function to update plot for each frame in the animation
def update_plot(frame, t, signals, lines, ax, is_freq=False, ylims=None):
    # Define the data range for this frame
    start, end = frame * chunk_size, (frame + 1) * chunk_size
    if end > len(signals[0]):  # Ensure we don't exceed signal length
        end = len(signals[0])
    
    for i, signal in enumerate(signals):
        if is_freq:  # If plotting in frequency domain
            spectrum = np.abs(np.fft.fftshift(np.fft.fft(signal[start:end])))  # Compute FFT
            freq = np.fft.fftshift(np.fft.fftfreq(len(spectrum), d=t[1] - t[0]))  # Frequency vector
            lines[i].set_data(freq, spectrum)  # Update frequency plot
            ax[i, 1].set_xlim(freq.min(), freq.max())
            ax[i, 1].set_ylim(1e-3, 1.1 * spectrum.max())
        else:  # If plotting in time domain
            lines[i].set_data(t[start:end], signal[start:end])  # Update time plot
            ax[i, 0].set_xlim(t[start], t[end])
            if ylims:  # Set y-limits if provided
                ax[i, 0].set_ylim(ylims)
            else:
                ax[i, 0].set_ylim(-1.1, 1.1)  # Default y-limits
    return lines

# Function to animate original and modulated signals in subplots
def animate_signals_in_subplots(t, signals, fs, window_title):
    fig, axs = plt.subplots(2, 2, figsize=(14, 8))  # Create a 2x2 grid of subplots
    fig.canvas.manager.set_window_title(window_title)  # Set window title
    
    # Set titles for each subplot
    axs[0, 0].set_title("Original Audio - Time Domain")
    axs[0, 1].set_title("Original Audio - Frequency Domain")
    axs[1, 0].set_title("SSB Modulated Signal - Time Domain")
    axs[1, 1].set_title("SSB Modulated Signal - Frequency Domain")

    # Initialize empty lines for animation
    time_lines = [axs[i, 0].plot([], [])[0] for i in range(2)]
    freq_lines = [axs[i, 1].plot([], [])[0] for i in range(2)]

    # Set axis labels
    for ax in axs.flat:
        ax.set_xlabel("Time (s)" if ax in axs[:, 0] else "Frequency (Hz)")
        ax.set_ylabel("Amplitude" if ax in axs[:, 0] else "Magnitude")
        ax.set_yscale('log' if ax in axs[:, 1] else 'linear')
    
    # Initialize function for animation
    def init():
        for line in time_lines + freq_lines:
            line.set_data([], [])
        return time_lines + freq_lines

    # Animation function
    def animate(frame):
        update_plot(frame, t, signals, time_lines, axs)  # Update time domain plots
        update_plot(frame, t, signals, freq_lines, axs, is_freq=True)  # Update frequency domain plots
        if frame * chunk_size >= len(t):  # Stop animation if end is reached
            anim.event_source.stop()
            plt.close(fig)  # Close the figure window when animation stops
        return time_lines + freq_lines

    # Create animation
    anim = FuncAnimation(fig, animate, init_func=init, frames=len(t) // chunk_size, interval=interval, blit=True)
    plt.tight_layout()
    plt.show()

# Function to animate the demodulated signal in separate plots
def animate_demodulated_signal(t, signal, fs, window_title):
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))  # Create a 1x2 grid of subplots
    fig.canvas.manager.set_window_title(window_title)  # Set window title

    # Set titles for each subplot
    axs[0].set_title("Demodulated Signal - Time Domain")
    axs[1].set_title("Demodulated Signal - Frequency Domain")

    # Initialize empty lines for animation
    time_line, = axs[0].plot([], [])
    freq_line, = axs[1].plot([], [])
    
    # Set axis labels and limits
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Amplitude")
    axs[0].set_ylim(-1.1, 1.1)
    
    axs[1].set_xlabel("Frequency (Hz)")
    axs[1].set_ylabel("Magnitude")
    axs[1].set_yscale('log')

    # Initialize function for animation
    def init():
        time_line.set_data([], [])
        freq_line.set_data([], [])
        return [time_line, freq_line]

    # Animation function
    def animate(frame):
        start, end = frame * chunk_size, (frame + 1) * chunk_size
        if end > len(signal):
            end = len(signal)
        # Update time domain plot
        time_line.set_data(t[start:end], signal[start:end])
        axs[0].set_xlim(t[start], t[end])
        
        # Update frequency domain plot
        spectrum = np.abs(np.fft.fftshift(np.fft.fft(signal[start:end])))
        freq = np.fft.fftshift(np.fft.fftfreq(len(spectrum), d=t[1] - t[0]))
        freq_line.set_data(freq, spectrum)
        axs[1].set_xlim(freq.min(), freq.max())
        axs[1].set_ylim(1e-3, 1.1 * spectrum.max())
        
        return [time_line, freq_line]

    # Create animation
    anim = FuncAnimation(fig, animate, init_func=init, frames=len(t) // chunk_size, interval=interval, blit=True)
    plt.tight_layout()
    plt.show()

# Function to plot static graphs
def plot_static_graphs(t, signals, demodulated_signal, fs):
    fig, axs = plt.subplots(3, 2, figsize=(14, 12))  # Create a 3x2 grid of subplots

    # Plot Original Audio - Time Domain
    axs[0, 0].plot(t, signals[0])
    axs[0, 0].set_title("Original Audio - Time Domain")
    axs[0, 0].set_ylabel("Amplitude")
    axs[0, 0].grid(True)  # Add grid lines
    axs[0, 0].set_xticklabels([])  # Remove x-axis labels

    # Plot Original Audio - Frequency Domain
    spectrum = np.abs(np.fft.fftshift(np.fft.fft(signals[0])))
    freq = np.fft.fftshift(np.fft.fftfreq(len(spectrum), d=t[1] - t[0]))
    axs[0, 1].plot(freq, spectrum)
    axs[0, 1].set_title("Original Audio - Frequency Domain")
    axs[0, 1].set_ylabel("Magnitude")
    axs[0, 1].set_yscale('log')
    axs[0, 1].grid(True)  # Add grid lines
    axs[0, 1].set_xticklabels([])  # Remove x-axis labels

    # Plot SSB Modulated Signal - Time Domain
    axs[1, 0].plot(t, signals[1])
    axs[1, 0].set_title("SSB Modulated Signal - Time Domain")
    axs[1, 0].set_ylabel("Amplitude")
    axs[1, 0].grid(True)  # Add grid lines
    axs[1, 0].set_xticklabels([])  # Remove x-axis labels

    # Plot SSB Modulated Signal - Frequency Domain
    spectrum = np.abs(np.fft.fftshift(np.fft.fft(signals[1])))
    freq = np.fft.fftshift(np.fft.fftfreq(len(spectrum), d=t[1] - t[0]))
    axs[1, 1].plot(freq, spectrum)
    axs[1, 1].set_title("SSB Modulated Signal - Frequency Domain")
    axs[1, 1].set_ylabel("Magnitude")
    axs[1, 1].set_yscale('log')
    axs[1, 1].grid(True)  # Add grid lines
    axs[1, 1].set_xticklabels([])  # Remove x-axis labels

    # Plot Demodulated Signal - Time Domain
    axs[2, 0].plot(t, demodulated_signal)
    axs[2, 0].set_title("Recovered Signal - Time Domain")
    axs[2, 0].set_xlabel("Time (s)")  # Add x-axis label
    axs[2, 0].set_ylabel("Amplitude")
    axs[2, 0].grid(True)  # Add grid lines

    # Plot Demodulated Signal - Frequency Domain
    spectrum = np.abs(np.fft.fftshift(np.fft.fft(demodulated_signal)))
    freq = np.fft.fftshift(np.fft.fftfreq(len(spectrum), d=t[1] - t[0]))
    axs[2, 1].plot(freq, spectrum)
    axs[2, 1].set_title("Recovered Signal - Frequency Domain")
    axs[2, 1].set_xlabel("Frequency (Hz)")  # Add x-axis label
    axs[2, 1].set_ylabel("Magnitude")
    axs[2, 1].set_yscale('log')
    axs[2, 1].grid(True)  # Add grid lines

    # Adjust layout to make space for x-axis labels
    plt.tight_layout(rect=[0, 0.05, 1, 1])  # Leave space at the bottom for x-axis labels
    plt.show()

# Load and preprocess the audio file
fs, audio = wavfile.read("audio2.wav")
if len(audio.shape) > 1:  # If stereo, select one channel
    audio = audio[:, 0]
audio = audio.astype(float) / np.max(np.abs(audio))  # Normalize audio
if len(audio) > 10 * fs:  # Limit audio length to 10 seconds
    audio = audio[:30 * fs]

t = np.arange(len(audio)) / fs  # Time vector
fc, N = 10000, 10001  # Carrier frequency and signal length

# Modulate and demodulate the signal
ssb_signal = ssb_modulate(audio, fc, fs, N)
demodulated_signal = ssb_demodulate(ssb_signal, fc, fs)

# Plot static graphs
plot_static_graphs(t, [audio, ssb_signal], demodulated_signal, fs)

# Play and animate the original and SSB modulated signals
print("Playing original audio and visualizing...")
sd.play(audio, fs)
animate_signals_in_subplots(t, [audio, ssb_signal], fs, "Original & SSB Modulated Signals")
sd.wait()

# Play and animate the demodulated signal
print("Playing recovered audio...")
sd.play(demodulated_signal, fs)
animate_demodulated_signal(t, demodulated_signal, fs, "Recovered Signal")
sd.wait()

# Save the recovered audio
wavfile.write("recovered_audio.wav", fs, (demodulated_signal * 32767).astype(np.int16))

#NOTE:Actually for the animated signals since the audio clip is of 10 secs, the plot is repeating in an infinite loop
#even after the audio stops.Close the window to see the other plots. Also the animated plots are shown for a particular snapshot
#of the time between 0 to 10 secs of the audio.So, it keeps changing after refreshing the window tab. 
