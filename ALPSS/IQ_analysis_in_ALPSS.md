# IQ Analysis in ALPSS

## Overview

**IQ (In-phase/Quadrature) analysis** is a key step in the ALPSS (Analysis of Photonic Doppler Velocimetry Signals of Spall) code. It is used to demodulate the raw PDV signal, extract the amplitude and phase, and—most importantly—automatically and robustly detect the start time of the physical event (e.g., spall) in the experiment.

---

## Where IQ Analysis Occurs

IQ analysis is implemented in the function `spall_doi_finder` in `alpss_main.py`. This function is called early in the main workflow by `alpss_main`.

---

## Purpose of IQ Analysis in ALPSS

- **Demodulate the signal**: Extracts amplitude and phase of the carrier frequency from the raw voltage signal.
- **Detect the signal start time**: Monitors the amplitude drop in the demodulated signal to identify when the physical event begins.

---

## How IQ Analysis is Performed

1. **Carrier Frequency Extraction**
   - Computes the FFT of the voltage signal.
   - Finds the frequency with the maximum amplitude within the user-specified frequency range (`freq_min` to `freq_max`).
   - This is taken as the carrier frequency.

2. **Demodulation**
   - The signal is demodulated using:
     - $I = V \cdot \cos(2\pi f_{carrier} t)$
     - $Q = V \cdot \sin(2\pi f_{carrier} t)$
     - Where $V$ is the voltage and $t$ is time.

3. **Smoothing**
   - Both I and Q are smoothed using a Gaussian window to reduce noise and avoid artifacts at the start of the signal.

4. **Amplitude and Phase Calculation**
   - Amplitude: $\sqrt{I^2 + Q^2}$
   - Phase: $\arctan2(Q, I)$ (unwrapped)

5. **Start Time Detection**
   - The initial amplitude is measured.
   - A threshold (e.g., 40% of the initial amplitude) is set.
   - The code finds the first time index where the amplitude drops below this threshold, which is interpreted as the event start time (`t_start_detected_iq`).

6. **Visualization**
   - The code plots the amplitude and phase, marking the detected start time for user inspection.

7. **Integration with Main Workflow**
   - The detected start time from IQ analysis (`t_start_detected_iq`) is used (sometimes in combination with other methods) to set the region of interest for further analysis (velocity extraction, spall analysis, etc.).

---

## How the IQ Results Are Used

- The IQ-detected start time (`t_start_detected_iq`) is stored in the output dictionary from `spall_doi_finder`.
- This value is used to set the time window for the main analysis, either directly or in combination with other start time detection methods.
- The amplitude and phase arrays from IQ analysis are also stored and can be used for further diagnostics or advanced analysis.

---

## Summary Table

| Step                | Purpose/Output                                  | How Used in ALPSS Workflow                |
|---------------------|-------------------------------------------------|-------------------------------------------|
| FFT of voltage      | Find carrier frequency                          | Used for demodulation                     |
| Demodulate (I, Q)   | Extract amplitude and phase                     | Used for start time detection             |
| Smoothing           | Reduce noise in I/Q                             | Improves robustness of amplitude drop     |
| Amplitude/Phase     | Calculate signal envelope and phase             | Amplitude drop used for event detection   |
| Start time detection| Find when amplitude drops below threshold       | Sets `t_start_detected_iq`                |
| Output to dict      | Store amplitude, phase, detected start time     | Used by rest of ALPSS pipeline            |

---

## Why Use IQ Analysis?

- **Robustness**: IQ demodulation is less sensitive to noise and phase shifts, making start time detection more reliable.
- **Physical Meaning**: The amplitude envelope directly reflects the presence/absence of the carrier, which is modulated by the physical event (e.g., spall).
- **Automation**: Enables automated, objective detection of the event start, which is critical for batch processing and reproducibility.

---

**In summary:**

**IQ analysis in ALPSS is used to robustly and automatically detect the start time of the physical event in PDV signals by demodulating the signal, extracting the amplitude envelope, and identifying a significant amplitude drop. This detected start time is then used to define the region of interest for all subsequent analysis steps.** 