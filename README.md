# SSB Modulation and Demodulation Implementation in Python

This Python script implements Single Sideband (SSB) Amplitude Modulation and Demodulation. It provides a comprehensive demonstration of the SSB process, including signal processing, visualization, and audio playback. Here's a breakdown of its main components and functionality:

## Key Components:

1. **SSB Modulation**: 
   - Uses the Hilbert transform to create an analytic signal.
   - Implements both upper and lower sideband modulation.

2. **SSB Demodulation**:
   - Reverses the modulation process.
   - Applies a low-pass filter to extract the baseband signal.

3. **Signal Visualization**:
   - Creates animated plots of signals in both time and frequency domains.
   - Generates static plots for comparison of original, modulated, and demodulated signals.

4. **Audio Processing**:
   - Loads and normalizes audio from a WAV file.
   - Plays back original and processed audio.

## Workflow:

1. Load and preprocess an audio file.
2. Perform SSB modulation on the audio signal.
3. Demodulate the SSB signal to recover the original audio.
4. Visualize the original, modulated, and demodulated signals using both static and animated plots.
5. Play back the original and recovered audio signals.
6. Save the recovered audio to a new WAV file.

## Technical Details:

- Uses NumPy for numerical operations and SciPy for signal processing.
- Employs Matplotlib for plotting and animation.
- Utilizes SoundDevice for audio playback.

## Notable Features:

- Real-time animation of signals in both time and frequency domains.
- Comparison of original, modulated, and demodulated signals.
- Audio playback synchronized with signal visualization.
- Comprehensive error handling and user feedback.

This implementation serves as an educational tool for understanding SSB modulation and demodulation, providing both visual and auditory feedback on the process.
