# %%
from datetime import datetime
import traceback
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to prevent windows
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd
import numpy as np
import os
from scipy.fft import fft, ifft, fftfreq
from scipy.fftpack import fftshift
from scipy.optimize import curve_fit
from scipy import signal
import findiff
import cv2 as cv

# Try to import ShortTimeFFT (available in scipy >= 1.9.0)
try:
    from scipy.signal import ShortTimeFFT
    SHORTTIMEFFT_AVAILABLE = True
except ImportError:
    # Fallback for older scipy versions
    SHORTTIMEFFT_AVAILABLE = False
    print("Warning: ShortTimeFFT not available in this scipy version. Using legacy STFT.")

from scipy.interpolate import interp1d
from scipy.signal import savgol_filter


def gaussian_window(window_length, std):
    n = np.arange(0, window_length) - (window_length - 1.0) / 2.0
    return np.exp(-0.5 * (n / std) ** 2)


# main function to link together all the sub-functions
def alpss_main(**inputs):
    print(f"[{datetime.now()}] DEBUG: alpss_main function called")
    print(f"[{datetime.now()}] DEBUG: save_data = {inputs.get('save_data', 'NOT FOUND')}")
    print(f"[{datetime.now()}] DEBUG: display_plots = {inputs.get('display_plots', 'NOT FOUND')}")

    # get the current working directory
    cwd = os.getcwd()

    # attempt to run the program in full
    try:

        # begin the program timer
        start_time = datetime.now()

        # function to find the spall signal domain of interest
        print(f"[{datetime.now()}] DEBUG: Starting spall_doi_finder...")
        sdf_out = spall_doi_finder(**inputs)
        print(f"[{datetime.now()}] DEBUG: spall_doi_finder complete.")

        # function to find the carrier frequency
        cen = carrier_frequency(sdf_out, **inputs)
        print(f"[{datetime.now()}] DEBUG: carrier_frequency complete.")

        # function to filter out the carrier frequency after the signal has started
        cf_out = carrier_filter(sdf_out, cen, **inputs)
        print(f"[{datetime.now()}] DEBUG: carrier_filter complete.")

        # function to calculate the velocity from the filtered voltage signal
        vc_out = velocity_calculation(sdf_out, cen, cf_out, **inputs)
        print(f"[{datetime.now()}] DEBUG: velocity_calculation complete.")

        # function to estimate the instantaneous uncertainty for all points in time
        iua_out = instantaneous_uncertainty_analysis(sdf_out, vc_out, cen, **inputs)
        print(f"[{datetime.now()}] DEBUG: instantaneous_uncertainty_analysis complete.")

        # function to find points of interest on the velocity trace
        sa_out = spall_analysis(vc_out, iua_out, **inputs)
        print(f"[{datetime.now()}] DEBUG: spall_analysis complete.")

        # function to calculate uncertainties in the spall strength and strain rate due to external uncertainties
        fua_out = full_uncertainty_analysis(cen, sa_out, iua_out, **inputs)
        print(f"[{datetime.now()}] DEBUG: full_uncertainty_analysis complete.")

        # end the program timer
        end_time = datetime.now()

        # function to generate the final figure - ORIGINAL ALPSS PLOT
        print(f"[{datetime.now()}] DEBUG: About to call original ALPSS plotting function...")
        fig = None
        
        # Check if user wants to save plots
        save_plots = inputs.get("save_all_plots", "no")
        if save_plots == "yes":
            try:
                fig = plotting(
                    sdf_out,
                    cen,
                    cf_out,
                    vc_out,
                    sa_out,
                    iua_out,
                    fua_out,
                    start_time,
                    end_time,
                    **inputs,
                )
                print(f"[{datetime.now()}] DEBUG: original ALPSS plotting completed successfully.")
            except Exception as e:
                print(f"[{datetime.now()}] ERROR in original ALPSS plotting: {e}\n{traceback.format_exc()}")
            
            # Also call simple_plotting to create individual plots in subfolder
            try:
                simple_plotting(
                    sdf_out,
                    cen,
                    cf_out,
                    vc_out,
                    sa_out,
                    iua_out,
                    fua_out,
                    start_time,
                    end_time,
                    **inputs,
                )
                print(f"[{datetime.now()}] DEBUG: simple_plotting completed successfully.")
            except Exception as e:
                print(f"[{datetime.now()}] ERROR in simple_plotting: {e}\n{traceback.format_exc()}")
        else:
            print(f"[{datetime.now()}] DEBUG: save_all_plots is not 'yes', skipping plot creation")

        # function to save the output files if desired
        if inputs["save_data"] == "yes":
            print(f"[{datetime.now()}] DEBUG: About to call saving function...")
            try:
                saving(
                    sdf_out,
                    cen,
                    vc_out,
                    sa_out,
                    iua_out,
                    fua_out,
                    start_time,
                    end_time,
                    fig,
                    **inputs,
                )
                print(f"[{datetime.now()}] DEBUG: saving function completed successfully.")
            except Exception as e:
                print(f"[{datetime.now()}] ERROR in saving: {e}\n{traceback.format_exc()}")
        else:
            print(f"[{datetime.now()}] DEBUG: save_data is not 'yes', skipping saving")

        # end final timer and display full runtime
        end_time2 = datetime.now()
        print(
            f"\nFull program runtime (including plotting and saving):\n{end_time2 - start_time}\n"
        )

    # in case the program throws an error
    except Exception:

        # print the traceback for the error
        print(f"[{datetime.now()}] ERROR: Exception in main pipeline:\n{traceback.format_exc()}")

        # attempt to plot the voltage signal from the imported data
        try:

            # import the desired data. Convert the time to skip and turn into number of rows
            t_step = 1 / inputs["sample_rate"]
            rows_to_skip = (
                inputs["header_lines"] + inputs["time_to_skip"] / t_step
            )  # skip the header lines too
            nrows = inputs["time_to_take"] / t_step

            # change directory to where the data is stored
            os.chdir(inputs["exp_data_dir"])
            data = pd.read_csv(
                inputs["filename"], skiprows=int(rows_to_skip), nrows=int(nrows)
            )

            # rename the columns of the data
            data.columns = ["Time", "Ampl"]

            # put the data into numpy arrays. Zero the time data
            time = data["Time"].to_numpy()
            time = time - time[0]
            voltage = data["Ampl"].to_numpy()

            # calculate the sample rate from the experimental data
            fs = 1 / np.mean(np.diff(time))

            # calculate the short time fourier transform
            f, t, Zxx = stft(voltage, fs, **inputs)

            # calculate magnitude of Zxx
            mag = np.abs(Zxx)

            # plotting
            fig, (ax1, ax2) = plt.subplots(1, 2, num=2, figsize=(11, 4), dpi=300)
            ax1.plot(time / 1e-9, voltage / 1e-3)
            ax1.set_xlabel("Time (ns)")
            ax1.set_ylabel("Voltage (mV)")
            ax2.imshow(
                10 * np.log10(mag**2),
                aspect="auto",
                origin="lower",
                interpolation="none",
                extent=[t[0] / 1e-9, t[-1] / 1e-9, f[0] / 1e9, f[-1] / 1e9],
                cmap=inputs["cmap"],
            )
            ax2.set_xlabel("Time (ns)")
            ax2.set_ylabel("Frequency (GHz)")
            fig.suptitle("ERROR: Program Failed", c="r", fontsize=16)

            plt.tight_layout()
            plt.savefig(f"{inputs['filename'][0:-4]}--error_plot.png", dpi=300, bbox_inches='tight')
            plt.close(fig)

        # if that also fails then print the traceback and stop running the program
        except Exception:
            print(f"[{datetime.now()}] ERROR: Exception in error plotting:\n{traceback.format_exc()}")

    # move back to the original working directory
    os.chdir(cwd)
    print(f"[{datetime.now()}] DEBUG: alpss_main function completed")


# function to filter out the carrier frequency
def carrier_filter(sdf_out, cen, **inputs):
    # unpack dictionary values in to individual variables
    time = sdf_out["time"]
    voltage = sdf_out["voltage"]
    t_start_corrected = sdf_out["t_start_corrected"]
    fs = sdf_out["fs"]
    order = inputs["order"]
    wid = inputs["wid"]
    f_min = inputs["freq_min"]
    f_max = inputs["freq_max"]
    t_doi_start = sdf_out["t_doi_start"]
    t_doi_end = sdf_out["t_doi_end"]
    
    # Check if notch filter should be used
    use_notch_filter = inputs.get("use_notch_filter", True)  # Default to True for backward compatibility
    
    if use_notch_filter:
        # get the index in the time array where the signal begins
        sig_start_idx = np.argmin(np.abs(time - t_start_corrected))

        # filter the data after the signal start time with a gaussian notch
        freq = fftshift(
            np.arange(-len(time[sig_start_idx:]) / 2, len(time[sig_start_idx:]) / 2)
            * fs
            / len(time[sig_start_idx:])
        )
        filt_2 = (
            1
            - np.exp(-((freq - cen) ** order) / wid**order)
            - np.exp(-((freq + cen) ** order) / wid**order)
        )
        voltage_filt = ifft(fft(voltage[sig_start_idx:]) * filt_2)

        # pair the filtered voltage from after the signal starts with the original data from before the signal starts
        voltage_filt = np.concatenate((voltage[0:sig_start_idx], voltage_filt))
    else:
        # Skip notch filtering - use original voltage
        voltage_filt = voltage
        print(f"[{datetime.now()}] INFO: Gaussian notch filter disabled - using unfiltered voltage signal")

    # perform a stft on the filtered voltage data. Only the real part as to not get a two sided spectrogram
    f_filt, t_filt, Zxx_filt = stft(np.real(voltage_filt), fs, **inputs)

    # calculate the power
    power_filt = 10 * np.log10(np.abs(Zxx_filt) ** 2)

    # cut the data to the domain of interest
    f_min_idx = np.argmin(np.abs(f_filt - f_min))
    f_max_idx = np.argmin(np.abs(f_filt - f_max))
    t_doi_start_idx = np.argmin(np.abs(t_filt - t_doi_start))
    t_doi_end_idx = np.argmin(np.abs(t_filt - t_doi_end))
    Zxx_filt_doi = Zxx_filt[f_min_idx:f_max_idx, t_doi_start_idx:t_doi_end_idx]
    power_filt_doi = power_filt[f_min_idx:f_max_idx, t_doi_start_idx:t_doi_end_idx]

    # save outputs to a dictionary
    cf_out = {
        "voltage_filt": voltage_filt,
        "f_filt": f_filt,
        "t_filt": t_filt,
        "Zxx_filt": Zxx_filt,
        "power_filt": power_filt,
        "Zxx_filt_doi": Zxx_filt_doi,
        "power_filt_doi": power_filt_doi,
        "notch_filter_used": use_notch_filter,
    }

    return cf_out


# calculate the carrier frequency as the frequency with the max amplitude within the frequency range of interest
# specified in the user inputs
def carrier_frequency(spall_doi_finder_outputs, **inputs):
    # unpack dictionary values in to individual variables
    fs = spall_doi_finder_outputs["fs"]
    time = spall_doi_finder_outputs["time"]
    voltage = spall_doi_finder_outputs["voltage"]
    freq_min = inputs["freq_min"]
    freq_max = inputs["freq_max"]

    # cut the time and voltage signals to only take the time during the user input "carrier_band_time".
    # That way there is none of the actual target signal in the FFT.
    # Need this because in some instances the target signal is stronger than the carrier, in which case the target signal may end up being filtered out.
    # These two lines of code should prevent that from happening and make sure the carrier is filtered properly.
    time = time[: int(round(inputs["carrier_band_time"] * fs))]
    voltage = voltage[: int(round(inputs["carrier_band_time"] * fs))]

    # calculate frequency values for fft
    freq = fftfreq(int(fs * time[-1]) + 1, 1 / fs)
    freq2 = freq[: int(freq.shape[0] / 2) - 1]

    # find the frequency indices that mark the range of interest
    freq_min_idx = np.argmin(np.abs(freq2 - freq_min))
    freq_max_idx = np.argmin(np.abs(freq2 - freq_max))

    # find the amplitude values for the fft
    ampl = np.abs(fft(voltage))
    ampl2 = ampl[: int(freq.shape[0] / 2) - 1]

    # cut the frequency and amplitude to the range of interest
    freq3 = freq2[freq_min_idx:freq_max_idx]
    ampl3 = ampl2[freq_min_idx:freq_max_idx]

    # find the carrier as the frequency with the max amplitude
    cen = freq3[np.argmax(ampl3)]

    # return the carrier frequency
    return cen


# program to calculate the uncertainty in the spall strength and strain rate
def full_uncertainty_analysis(cen, sa_out, iua_out, **inputs):
    """
    Based on the work of Mallick et al.

    Mallick, D.D., Zhao, M., Parker, J. et al. Laser-Driven Flyers and Nanosecond-Resolved Velocimetry for Spall Studies
    in Thin Metal Foils. Exp Mech 59, 611–628 (2019). https://doi.org/10.1007/s11340-019-00519-x
    """

    # unpack dictionary values in to individual variables
    rho = inputs["density"]
    C0 = inputs["C0"]
    lam = inputs["lam"]
    delta_rho = inputs["delta_rho"]
    delta_C0 = inputs["delta_C0"]
    delta_lam = inputs["delta_lam"]
    theta = inputs["theta"]
    delta_theta = inputs["delta_theta"]
    delta_freq_tb = sa_out["peak_velocity_freq_uncert"]
    delta_freq_td = sa_out["max_ten_freq_uncert"]
    delta_time_c = iua_out["tau"]
    delta_time_d = iua_out["tau"]
    freq_tb = (sa_out["v_max_comp"] * 2) / lam + cen
    freq_td = (sa_out["v_max_ten"] * 2) / lam + cen
    time_c = sa_out["t_max_comp"]
    time_d = sa_out["t_max_ten"]

    # assuming time c is the same as time b
    freq_tc = freq_tb
    delta_freq_tc = delta_freq_tb

    # convert angles to radians
    theta = theta * (np.pi / 180)
    delta_theta = delta_theta * (np.pi / 180)

    # calculate the individual terms for spall uncertainty
    term1 = (
        -0.5
        * rho
        * C0
        * (lam / 2)
        * np.tan(theta)
        * (1 / np.cos(theta))
        * (freq_tb - freq_td)
        * delta_theta
    )
    term2 = 0.5 * rho * C0 * (lam / (2 * np.cos(theta))) * delta_freq_tb
    term3 = -0.5 * rho * C0 * (lam / (2 * np.cos(theta))) * delta_freq_td
    term4 = 0.5 * rho * C0 * (1 / (2 * np.cos(theta))) * (freq_tb - freq_td) * delta_lam
    term5 = 0.5 * rho * (lam / (2 * np.cos(theta))) * (freq_tb - freq_td) * delta_C0
    term6 = 0.5 * C0 * (lam / (2 * np.cos(theta))) * (freq_tb - freq_td) * delta_rho

    # calculate spall uncertainty
    delta_spall = np.sqrt(
        term1**2 + term2**2 + term3**2 + term4**2 + term5**2 + term6**2
    )

    # calculate the individual terms for strain rate uncertainty
    d_f = freq_tc - freq_td
    d_t = time_d - time_c
    term7 = (-lam / (4 * C0**2 * np.cos(theta))) * (d_f / d_t) * delta_C0
    term8 = (1 / (4 * C0 * np.cos(theta))) * (d_f / d_t) * delta_lam
    term9 = (
        ((lam * np.tan(theta)) / (4 * C0 * np.cos(theta))) * (d_f / d_t) * delta_theta
    )
    term10 = (lam / (4 * C0 * np.cos(theta))) * (1 / d_t) * delta_freq_tc
    term11 = (-lam / (4 * C0 * np.cos(theta))) * (1 / d_t) * delta_freq_td
    term12 = (-lam / (4 * C0 * np.cos(theta))) * (d_f / d_t**2) * delta_time_c
    term13 = (lam / (4 * C0 * np.cos(theta))) * (d_f / d_t**2) * delta_time_d

    # calculate strain rate uncertainty
    delta_strain_rate = np.sqrt(
        term7**2 + term8**2 + term9**2 + term10**2 + term11**2 + term12**2 + term13**2
    )

    # save outputs to a dictionary
    fua_out = {"spall_uncert": delta_spall, "strain_rate_uncert": delta_strain_rate}

    return fua_out


# general function for a sinusoid
def sin_func(x, a, b, c, d):
    return a * np.sin(2 * np.pi * b * x + c) + d


# get the indices for the upper and lower envelope of the voltage signal
# https://stackoverflow.com/questions/34235530/how-to-get-high-and-low-envelope-of-a-signal
def hl_envelopes_idx(s, dmin=1, dmax=1, split=False):
    """
    Input :
    s: 1d-array, data signal from which to extract high and low envelopes
    dmin, dmax: int, optional, size of chunks, use this if the size of the input signal is too big
    split: bool, optional, if True, split the signal in half along its mean, might help to generate the envelope in some cases
    Output :
    lmin,lmax : high/low envelope idx of input signal s
    """

    # locals min
    lmin = (np.diff(np.sign(np.diff(s))) > 0).nonzero()[0] + 1
    # locals max
    lmax = (np.diff(np.sign(np.diff(s))) < 0).nonzero()[0] + 1

    if split:
        # s_mid is zero if s centered around x-axis or more generally mean of signal
        s_mid = np.mean(s)
        # pre-sorting of locals min based on relative position with respect to s_mid
        lmin = lmin[s[lmin] < s_mid]
        # pre-sorting of local max based on relative position with respect to s_mid
        lmax = lmax[s[lmax] > s_mid]

    # global min of dmin-chunks of locals min
    lmin = lmin[
        [i + np.argmin(s[lmin[i : i + dmin]]) for i in range(0, len(lmin), dmin)]
    ]
    # global max of dmax-chunks of locals max
    lmax = lmax[
        [i + np.argmax(s[lmax[i : i + dmax]]) for i in range(0, len(lmax), dmax)]
    ]

    return lmin, lmax


# gaussian distribution
def gauss(x, amp, sigma, mu):
    f = (amp / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    return f


# calculate the fwhm of a gaussian distribution
def fwhm(
    smoothing_window, smoothing_wid, smoothing_amp, smoothing_sigma, smoothing_mu, fs
):
    # x points for the gaussian weights
    x = np.linspace(-smoothing_wid, smoothing_wid, smoothing_window)

    # calculate the gaussian weights
    weights = gauss(x, smoothing_amp, smoothing_sigma, smoothing_mu)

    # calculate the half max
    half_max = ((np.max(weights) - np.min(weights)) / 2) + np.min(weights)

    # calculate the fwhm of the gaussian weights for the normalized x points
    fwhm_norm = 2 * np.abs(x[np.argmin(np.abs(weights - half_max))])

    # scale the fwhm to the number of points being used for the smoothing window
    fwhm_pts = (fwhm_norm / (smoothing_wid * 2)) * smoothing_window

    # calculate the time span of the fwhm of the gaussian weights
    fwhm = fwhm_pts / fs

    return fwhm


# function to estimate the instantaneous uncertainty for all points in time
def instantaneous_uncertainty_analysis(sdf_out, vc_out, cen, **inputs):
    # unpack needed variables
    lam = inputs["lam"]
    smoothing_window = inputs["smoothing_window"]
    smoothing_wid = inputs["smoothing_wid"]
    smoothing_amp = inputs["smoothing_amp"]
    smoothing_sigma = inputs["smoothing_sigma"]
    smoothing_mu = inputs["smoothing_mu"]
    fs = sdf_out["fs"]
    time = sdf_out["time"]
    time_f = vc_out["time_f"]
    voltage_filt = vc_out["voltage_filt"]
    time_start_idx = vc_out["time_start_idx"]
    time_end_idx = vc_out["time_end_idx"]
    carrier_band_time = inputs["carrier_band_time"]

    # take only real component of the filtered voltage signal
    voltage_filt = np.real(voltage_filt)

    # amount of time from the beginning of the voltage signal to analyze for noise
    t_take = carrier_band_time
    steps_take = int(t_take * fs)

    # get the data for only the beginning section of the signal
    time_cut = time[0:steps_take]
    voltage_filt_early = voltage_filt[0:steps_take]

    try:
        # fit a sinusoid to the data
        popt, pcov = curve_fit(
            sin_func, time_cut, voltage_filt_early, p0=[0.1, cen, 0, 0]
        )
    except Exception:
        # if sin fitting doesn't work set the fitting parameters to be zeros
        print(traceback.format_exc())
        popt = [0, 0, 0, 0]
        pcov = [0, 0, 0, 0]

    # calculate the fitted curve
    volt_fit = sin_func(time_cut, popt[0], popt[1], popt[2], popt[3])
    # print(popt[1])
    # print(cen)

    # calculate the residuals
    noise = voltage_filt_early - volt_fit

    # get data for only the doi of the voltage
    voltage_filt_doi = voltage_filt[time_start_idx:time_end_idx]

    # calculate the envelope indices of the originally imported voltage data (and now filtered) using the stack
    # overflow code
    lmin, lmax = hl_envelopes_idx(voltage_filt_doi, dmin=1, dmax=1, split=False)

    # interpolate the voltage envelope to every time point
    env_max_interp = np.interp(time_f, time_f[lmax], voltage_filt_doi[lmax])
    env_min_interp = np.interp(time_f, time_f[lmin], voltage_filt_doi[lmin])

    # calculate the estimated peak to peak amplitude at every time
    inst_amp = env_max_interp - env_min_interp

    # calculate the estimated noise fraction at every time
    # https://doi.org/10.1063/12.0000870
    inst_noise = np.std(noise) / (inst_amp / 2)

    # calculate the frequency and velocity uncertainty
    # https://doi.org/10.1063/12.0000870
    # take the characteristic time to be the fwhm of the gaussian weights used for smoothing the velocity signal
    tau = fwhm(
        smoothing_window,
        smoothing_wid,
        smoothing_amp,
        smoothing_sigma,
        smoothing_mu,
        fs,
    )
    freq_uncert_scaling = (1 / np.pi) * (np.sqrt(6 / (fs * (tau**3))))
    freq_uncert = inst_noise * freq_uncert_scaling
    vel_uncert = freq_uncert * (lam / 2)

    # dictionary to return outputs
    iua_out = {
        "time_cut": time_cut,
        "popt": popt,
        "pcov": pcov,
        "volt_fit": volt_fit,
        "noise": noise,
        "env_max_interp": env_max_interp,
        "env_min_interp": env_min_interp,
        "inst_amp": inst_amp,
        "inst_noise": inst_noise,
        "tau": tau,
        "freq_uncert_scaling": freq_uncert_scaling,
        "freq_uncert": freq_uncert,
        "vel_uncert": vel_uncert,
    }

    return iua_out


# function to take the numerical derivative of input array phas (central difference with a 9-point stencil).
# phas is padded so that after smoothing the final velocity trace matches the length of the domain of interest.
# this avoids issues with handling the boundaries in the derivative and later in smoothing.
# https://github.com/maroba/findiff/tree/master
def num_derivative(phas, window, time_start_idx, time_end_idx, fs):
    # set 8th order accuracy to get a 9-point stencil. can change the accuracy order if desired
    acc = 8

    # calculate how much padding is needed. half_space padding comes from the length of the smoothing window.
    half_space = int(np.floor(window / 2))
    pad = int(half_space + acc / 2)

    # get only the section of interest
    phas_pad = phas[time_start_idx - pad : time_end_idx + pad]

    # calculate the phase angle derivative
    ddt = findiff.FinDiff(0, 1 / fs, 1, acc=acc)
    dpdt_pad = ddt(phas_pad) * (1 / (2 * np.pi))

    # this is the hard coded 9-point central difference code. this can be used in case the findiff package ever breaks
    # dpdt_pad = np.zeros(phas_pad.shape)
    # for i in range(4, len(dpdt_pad) - 4):
    #     dpdt_pad[i] = ((1 / 280) * phas_pad[i - 4]
    #                    + (-4 / 105) * phas_pad[i - 3]
    #                    + (1 / 5) * phas_pad[i - 2]
    #                    + (-4 / 5) * phas_pad[i - 1]
    #                    + (4 / 5) * phas_pad[i + 1]
    #                    + (-1 / 5) * phas_pad[i + 2]
    #                    + (4 / 105) * phas_pad[i + 3]
    #                    + (-1 / 280) * phas_pad[i + 4]) \
    #                   * (fs / (2 * np.pi))

    # output both the padded and un-padded derivatives
    dpdt = dpdt_pad[pad:-pad]
    dpdt_pad = dpdt_pad[int(acc / 2) : -int(acc / 2)]

    return dpdt, dpdt_pad


# function to generate the final figure
def plotting(
    sdf_out,
    cen,
    cf_out,
    vc_out,
    sa_out,
    iua_out,
    fua_out,
    start_time,
    end_time,
    **inputs,
):
    # create the figure and axes
    fig = plt.figure(num=1, figsize=inputs["plot_figsize"], dpi=inputs["plot_dpi"])
    ax1 = plt.subplot2grid((3, 5), (0, 0))  # voltage data
    ax2 = plt.subplot2grid((3, 5), (0, 1))  # noise distribution histogram
    ax3 = plt.subplot2grid((3, 5), (1, 0))  # imported voltage spectrogram
    ax4 = plt.subplot2grid((3, 5), (1, 1))  # thresholded spectrogram
    ax5 = plt.subplot2grid((3, 5), (2, 0))  # spectrogram of the ROI
    ax6 = plt.subplot2grid((3, 5), (2, 1))  # filtered spectrogram of the ROI
    ax7 = plt.subplot2grid((3, 5), (0, 2), colspan=2)  # voltage in the ROI
    ax8 = plt.subplot2grid(
        (3, 5), (1, 2), colspan=2, rowspan=2
    )  # velocity overlaid with spectrogram
    ax9 = ax8.twinx()  # spectrogram overlaid with velocity
    ax10 = plt.subplot2grid((3, 5), (0, 4))  # noise fraction
    ax11 = ax10.twinx()  # velocity uncertainty
    ax12 = plt.subplot2grid((3, 5), (1, 4))  # velocity trace and spall points
    ax13 = plt.subplot2grid((3, 5), (2, 4), colspan=1, rowspan=1)  # results table

    # voltage data
    ax1.plot(
        sdf_out["time"] / 1e-9,
        sdf_out["voltage"] * 1e3,
        label="Original Signal",
        c="tab:blue",
    )
    ax1.plot(
        sdf_out["time"] / 1e-9,
        np.real(vc_out["voltage_filt"]) * 1e3,
        label="Filtered Signal",
        c="tab:orange",
    )
    ax1.plot(
        iua_out["time_cut"] / 1e-9,
        iua_out["volt_fit"] * 1e3,
        label="Sine Fit",
        c="tab:green",
    )
    ax1.axvspan(
        sdf_out["t_doi_start"] / 1e-9,
        sdf_out["t_doi_end"] / 1e-9,
        ymin=-1,
        ymax=1,
        color="tab:red",
        alpha=0.35,
        ec="none",
        label="ROI",
        zorder=4,
    )
    ax1.set_xlabel("Time (ns)")
    ax1.set_ylabel("Voltage (mV)")
    ax1.set_xlim([sdf_out["time"][0] / 1e-9, sdf_out["time"][-1] / 1e-9])
    ax1.legend(loc="upper right")
    ax1.set_title("Voltage Data")

    # noise distribution histogram
    ax2.hist(iua_out["noise"] * 1e3, bins=50, rwidth=0.8)
    ax2.set_xlabel("Noise (mV)")
    ax2.set_ylabel("Counts")
    ax2.set_title("Voltage Noise")

    # imported voltage spectrogram and a rectangle to show the ROI
    plt3 = ax3.imshow(
        10 * np.log10(sdf_out["mag"] ** 2),
        aspect="auto",
        origin="lower",
        interpolation="none",
        extent=[
            sdf_out["t"][0] / 1e-9,
            sdf_out["t"][-1] / 1e-9,
            sdf_out["f"][0] / 1e9,
            sdf_out["f"][-1] / 1e9,
        ],
        cmap=inputs["cmap"],
    )
    fig.colorbar(plt3, ax=ax3, label="Power (dBm)")
    anchor = [sdf_out["t_doi_start"] / 1e-9, sdf_out["f_doi"][0] / 1e9]
    width = sdf_out["t_doi_end"] / 1e-9 - sdf_out["t_doi_start"] / 1e-9
    height = sdf_out["f_doi"][-1] / 1e9 - sdf_out["f_doi"][0] / 1e9
    win = Rectangle(
        anchor,
        width,
        height,
        edgecolor="r",
        facecolor="none",
        linewidth=0.75,
        linestyle="-",
    )
    ax3.add_patch(win)
    ax3.set_xlabel("Time (ns)")
    ax3.set_ylabel("Frequency (GHz)")
    ax3.minorticks_on()
    ax3.set_title("Spectrogram Original Signal")

    # plotting the thresholded spectrogram on the ROI to show how the signal start time is found
    ax4.imshow(
        sdf_out["th3"],
        aspect="auto",
        origin="lower",
        interpolation="none",
        extent=[
            sdf_out["t"][0] / 1e-9,
            sdf_out["t"][-1] / 1e-9,
            sdf_out["f_doi"][0] / 1e9,
            sdf_out["f_doi"][-1] / 1e9,
        ],
        cmap=inputs["cmap"],
    )
    ax4.axvline(sdf_out["t_start_detected"] / 1e-9, ls="--", c="r")
    ax4.axvline(sdf_out["t_start_corrected"] / 1e-9, ls="-", c="r")
    if inputs["start_time_user"] == "none":
        ax4.axhline(sdf_out["f_doi"][sdf_out["f_doi_carr_top_idx"]] / 1e9, c="r")
    ax4.set_ylim([inputs["freq_min"] / 1e9, inputs["freq_max"] / 1e9])
    ax4.set_xlim([sdf_out["t_doi_start"] / 1e-9, sdf_out["t_doi_end"] / 1e-9])
    ax4.set_xlabel("Time (ns)")
    ax4.set_ylabel("Frequency (GHz)")
    ax4.minorticks_on()
    ax4.set_title("Thresholded Spectrogram")

    # plotting the spectrogram of the ROI with the start-time line to see how well it lines up
    plt5 = ax5.imshow(
        10 * np.log10(sdf_out["mag"] ** 2),
        aspect="auto",
        origin="lower",
        interpolation="none",
        extent=[
            sdf_out["t"][0] / 1e-9,
            sdf_out["t"][-1] / 1e-9,
            sdf_out["f"][0] / 1e9,
            sdf_out["f"][-1] / 1e9,
        ],
        cmap=inputs["cmap"],
    )
    fig.colorbar(plt5, ax=ax5, label="Power (dBm)")
    ax5.axvline(sdf_out["t_start_detected"] / 1e-9, ls="--", c="r")
    ax5.axvline(sdf_out["t_start_corrected"] / 1e-9, ls="-", c="r")
    if inputs["start_time_user"] == "none":
        ax5.axhline(sdf_out["f_doi"][sdf_out["f_doi_carr_top_idx"]] / 1e9, c="r")
    ax5.set_ylim([inputs["freq_min"] / 1e9, inputs["freq_max"] / 1e9])
    ax5.set_xlim([sdf_out["t_doi_start"] / 1e-9, sdf_out["t_doi_end"] / 1e-9])
    plt5.set_clim([np.min(sdf_out["power_doi"]), np.max(sdf_out["power_doi"])])
    ax5.set_xlabel("Time (ns)")
    ax5.set_ylabel("Frequency (GHz)")
    ax5.minorticks_on()
    ax5.set_title("Spectrogram ROI")

    # plotting the filtered spectrogram of the ROI
    plt6 = ax6.imshow(
        cf_out["power_filt"],
        aspect="auto",
        origin="lower",
        interpolation="none",
        extent=[
            cf_out["t_filt"][0] / 1e-9,
            cf_out["t_filt"][-1] / 1e-9,
            cf_out["f_filt"][0] / 1e9,
            cf_out["f_filt"][-1] / 1e9,
        ],
        cmap=inputs["cmap"],
    )
    fig.colorbar(plt6, ax=ax6, label="Power (dBm)")
    ax6.axvline(sdf_out["t_start_detected"] / 1e-9, ls="--", c="r")
    ax6.axvline(sdf_out["t_start_corrected"] / 1e-9, ls="-", c="r")
    ax6.set_ylim([inputs["freq_min"] / 1e9, inputs["freq_max"] / 1e9])
    ax6.set_xlim([sdf_out["t_doi_start"] / 1e-9, sdf_out["t_doi_end"] / 1e-9])
    plt6.set_clim([np.min(cf_out["power_filt_doi"]), np.max(cf_out["power_filt_doi"])])
    ax6.set_xlabel("Time (ns)")
    ax6.set_ylabel("Frequency (GHz)")
    ax6.minorticks_on()
    ax6.set_title("Filtered Spectrogram ROI")

    # voltage in the ROI and the signal envelope
    ax7.plot(
        sdf_out["time"] / 1e-9,
        np.real(vc_out["voltage_filt"]) * 1e3,
        label="Filtered Signal",
        c="tab:blue",
    )
    ax7.plot(
        vc_out["time_f"] / 1e-9,
        iua_out["env_max_interp"] * 1e3,
        label="Signal Envelope",
        c="tab:red",
    )
    ax7.plot(vc_out["time_f"] / 1e-9, iua_out["env_min_interp"] * 1e3, c="tab:red")
    ax7.set_xlabel("Time (ns)")
    ax7.set_ylabel("Voltage (mV)")
    ax7.set_xlim([sdf_out["t_doi_start"] / 1e-9, sdf_out["t_doi_end"] / 1e-9])
    ax7.legend(loc="upper right")
    ax7.set_title("Voltage ROI")

    # plotting the velocity and smoothed velocity curves to be overlaid on top of the spectrogram
    ax8.plot(
        (vc_out["time_f"]) / 1e-9,
        vc_out["velocity_f"],
        "-",
        c="grey",
        alpha=0.65,
        linewidth=3,
        label="Velocity",
    )
    ax8.plot(
        (vc_out["time_f"]) / 1e-9,
        vc_out["velocity_f_smooth"],
        "k-",
        linewidth=3,
        label="Smoothed Velocity",
    )
    ax8.plot(
        vc_out["time_f"] / 1e-9,
        vc_out["velocity_f_smooth"] + iua_out["vel_uncert"] * inputs["uncert_mult"],
        "r-",
        alpha=0.5,
        label=rf'$1\sigma$ Uncertainty (multiplied by {inputs["uncert_mult"]})',
    )
    ax8.plot(
        vc_out["time_f"] / 1e-9,
        vc_out["velocity_f_smooth"] - iua_out["vel_uncert"] * inputs["uncert_mult"],
        "r-",
        alpha=0.5,
    )
    ax8.set_xlabel("Time (ns)")
    ax8.set_ylabel("Velocity (m/s)")
    ax8.legend(loc="lower right", fontsize=9, framealpha=1)
    ax8.set_zorder(1)
    ax8.patch.set_visible(False)
    ax8.set_title("Filtered Spectrogram ROI with Velocity")

    # plotting the final spectrogram to go with the velocity curves
    plt9 = ax9.imshow(
        cf_out["power_filt"],
        extent=[
            cf_out["t_filt"][0] / 1e-9,
            cf_out["t_filt"][-1] / 1e-9,
            cf_out["f_filt"][0] / 1e9,
            cf_out["f_filt"][-1] / 1e9,
        ],
        aspect="auto",
        origin="lower",
        interpolation="none",
        cmap=inputs["cmap"],
    )
    ax9.set_ylabel("Frequency (GHz)")
    vel_lim = np.array([-300, np.max(vc_out["velocity_f_smooth"]) + 300])
    ax8.set_ylim(vel_lim)
    ax8.set_xlim([cf_out["t_filt"][0] / 1e-9, cf_out["t_filt"][-1] / 1e-9])
    freq_lim = (vel_lim / (inputs["lam"] / 2)) + cen
    ax9.set_ylim(freq_lim / 1e9)
    ax9.set_xlim([sdf_out["t_doi_start"] / 1e-9, sdf_out["t_doi_end"] / 1e-9])
    ax9.minorticks_on()
    plt9.set_clim([np.min(cf_out["power_filt_doi"]), np.max(cf_out["power_filt_doi"])])

    # plot the noise fraction on the ROI
    ax10.plot(vc_out["time_f"] / 1e-9, iua_out["inst_noise"] * 100, "r", linewidth=2)
    ax10.set_xlabel("Time (ns)")
    ax10.set_ylabel("Noise Fraction (%)")
    ax10.set_xlim([vc_out["time_f"][0] / 1e-9, vc_out["time_f"][-1] / 1e-9])
    ax10.minorticks_on()
    ax10.grid(axis="both", which="both")
    ax10.set_title("Noise Fraction and Velocity Uncertainty")

    # plot the velocity uncertainty on the ROI
    ax11.plot(vc_out["time_f"] / 1e-9, iua_out["vel_uncert"], linewidth=2)
    ax11.set_ylabel("Velocity Uncertainty (m/s)")
    ax11.minorticks_on()

    # plotting the final smoothed velocity trace and uncertainty bounds with spall point markers (if they were found
    # on the signal)
    ax12.fill_between(
        (vc_out["time_f"] - sdf_out["t_start_corrected"]) / 1e-9,
        vc_out["velocity_f_smooth"] + 2 * iua_out["vel_uncert"] * inputs["uncert_mult"],
        vc_out["velocity_f_smooth"] - 2 * iua_out["vel_uncert"] * inputs["uncert_mult"],
        color="mistyrose",
        label=rf'$2\sigma$ Uncertainty (multiplied by {inputs["uncert_mult"]})',
    )

    ax12.fill_between(
        (vc_out["time_f"] - sdf_out["t_start_corrected"]) / 1e-9,
        vc_out["velocity_f_smooth"] + iua_out["vel_uncert"] * inputs["uncert_mult"],
        vc_out["velocity_f_smooth"] - iua_out["vel_uncert"] * inputs["uncert_mult"],
        color="lightcoral",
        alpha=0.5,
        ec="none",
        label=rf'$1\sigma$ Uncertainty (multiplied by {inputs["uncert_mult"]})',
    )

    ax12.plot(
        (vc_out["time_f"] - sdf_out["t_start_corrected"]) / 1e-9,
        vc_out["velocity_f_smooth"],
        "k-",
        linewidth=3,
        label="Smoothed Velocity",
    )
    ax12.set_xlabel("Time (ns)")
    ax12.set_ylabel("Velocity (m/s)")
    ax12.set_title("Velocity with Uncertainty Bounds")

    if not np.isnan(sa_out["t_max_comp"]):
        ax12.plot(
            (sa_out["t_max_comp"] - sdf_out["t_start_corrected"]) / 1e-9,
            sa_out["v_max_comp"],
            "bs",
            label=f'Velocity at Max Compression: {int(round(sa_out["v_max_comp"]))}',
        )
    if not np.isnan(sa_out["t_max_ten"]):
        ax12.plot(
            (sa_out["t_max_ten"] - sdf_out["t_start_corrected"]) / 1e-9,
            sa_out["v_max_ten"],
            "ro",
            label=f'Velocity at Max Tension: {int(round(sa_out["v_max_ten"]))}',
        )
    if not np.isnan(sa_out["t_rc"]):
        ax12.plot(
            (sa_out["t_rc"] - sdf_out["t_start_corrected"]) / 1e-9,
            sa_out["v_rc"],
            "gD",
            label=f'Velocity at Recompression: {int(round(sa_out["v_rc"]))}',
        )

    # if not np.isnan(sa_out['t_max_comp']) or not np.isnan(sa_out['t_max_ten']) or not np.isnan(sa_out['t_rc']):
    #    ax12.legend(loc='lower right', fontsize=9)
    ax12.legend(loc="lower right", fontsize=9)
    ax12.set_xlim(
        [
            -inputs["t_before"] / 1e-9,
            (vc_out["time_f"][-1] - sdf_out["t_start_corrected"]) / 1e-9,
        ]
    )
    ax12.set_ylim(
        [
            np.min(vc_out["velocity_f_smooth"]) - 100,
            np.max(vc_out["velocity_f_smooth"]) + 100,
        ]
    )

    if np.max(iua_out["inst_noise"]) > 1.0:
        ax10.set_ylim([0, 100])
        ax11.set_ylim([0, iua_out["freq_uncert_scaling"] * (inputs["lam"] / 2)])

    # table to show results of the run
    run_data1 = {
        "Name": [
            "Date",
            "Time",
            "File Name",
            "Run Time",
            "Smoothing FWHM (ns)",
            "Peak Shock Stress (GPa)",
            "Strain Rate (x1e6)",
            "Spall Strength (GPa)",
        ],
        "Value": [
            start_time.strftime("%b %d %Y"),
            start_time.strftime("%I:%M %p"),
            inputs["filename"],
            (end_time - start_time),
            round(iua_out["tau"] * 1e9, 2),
            round(
                (0.5 * inputs["density"] * inputs["C0"] * sa_out["v_max_comp"]) / 1e9, 6
            ),
            rf"{round(sa_out['strain_rate_est'] / 1e6, 6)} $\pm$ {round(fua_out['strain_rate_uncert'] / 1e6, 6)}",
            rf"{round(sa_out['spall_strength_est'] / 1e9, 6)} $\pm$ {round(fua_out['spall_uncert'] / 1e9, 6)}",
        ],
    }

    df1 = pd.DataFrame(data=run_data1)
    cellLoc1 = "center"
    loc1 = "center"
    table1 = ax13.table(
        cellText=df1.values, colLabels=df1.columns, cellLoc=cellLoc1, loc=loc1
    )
    table1.auto_set_font_size(False)
    table1.set_fontsize(10)
    table1.scale(1, 1.5)
    ax13.axis("tight")
    ax13.axis("off")

    # fix the layout
    plt.tight_layout()

    # Save each subplot as a separate figure with IQ-style formatting
    subplot_info = [
        (ax1, 'voltage_data'),
        (ax2, 'noise_histogram'),
        (ax3, 'imported_spectrogram'),
        (ax4, 'thresholded_spectrogram'),
        (ax5, 'roi_spectrogram'),
        (ax6, 'filtered_roi_spectrogram'),
        (ax7, 'voltage_roi'),
        (ax8, 'velocity_spectrogram_overlay'),
        (ax10, 'noise_fraction'),
        (ax11, 'velocity_uncertainty'),
        (ax12, 'velocity_trace_spall'),
        (ax13, 'results_table'),
    ]
    # Output directory and filename prefix
    out_dir = inputs.get('out_files_dir', '.')
    fname_prefix = os.path.splitext(inputs.get('filename', 'ALPSS'))[0]
    for ax, tag in subplot_info:
        fig_sub, ax_sub = plt.subplots(figsize=(8, 8))
        for line in ax.get_lines():
            ax_sub.plot(line.get_xdata(), line.get_ydata(), label=line.get_label(), color=line.get_color(), linestyle=line.get_linestyle())
        # Copy images if present (for imshow plots)
        for im in ax.get_images():
            ax_sub.imshow(im.get_array(), aspect='auto', origin=im.origin, extent=im.get_extent(), cmap=im.get_cmap())
        # Copy fill_between if present
        # Commented out due to matplotlib error - can't add same collection to multiple figures
        # for collection in ax.collections:
        #     ax_sub.add_collection(collection)
        # Copy axis labels and title
        ax_sub.set_xlabel(ax.get_xlabel(), fontsize=20)
        ax_sub.set_ylabel(ax.get_ylabel(), fontsize=20)
        ax_sub.set_title(ax.get_title(), fontsize=20)
        # Copy axis limits
        ax_sub.set_xlim(ax.get_xlim())
        ax_sub.set_ylim(ax.get_ylim())
        # Copy ticks
        ax_sub.tick_params(axis='both', labelsize=20)
        # Copy legend if present
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax_sub.legend(handles, labels, fontsize=12)
        plt.tight_layout()
        fig_sub.savefig(os.path.join(out_dir, f"{fname_prefix}--{tag}.png"), dpi=inputs.get('plot_dpi', 300), format='png', facecolor='w')
        plt.close(fig_sub)

    # display the plots if desired. if this is turned off the plots will still save
    # Disabled plot display to prevent hanging
    # if inputs["display_plots"] == "yes":
    #     plt.show()

    # return the figure so it can be saved if desired
    return fig


# Simplified plotting function that avoids matplotlib artist copying issues
def simple_plotting(
    sdf_out,
    cen,
    cf_out,
    vc_out,
    sa_out,
    iua_out,
    fua_out,
    start_time,
    end_time,
    **inputs,
):
    print(f"[{datetime.now()}] Creating simplified plots...")
    os.makedirs(inputs["out_files_dir"], exist_ok=True)
    
    # Check if user wants to save all plots in subfolder
    save_all_plots = inputs.get("save_all_plots", "no")
    base_filename = inputs["filename"][0:-4]  # Remove file extension
    
    if save_all_plots == "yes":
        # Create subfolder for this file's plots
        plots_subfolder = os.path.join(inputs["out_files_dir"], f"{base_filename}_plots")
        os.makedirs(plots_subfolder, exist_ok=True)
        print(f"[{datetime.now()}] Saving all plots in subfolder: {plots_subfolder}")
        plot_dir = plots_subfolder
    else:
        # Save plots in main output directory
        plot_dir = inputs["out_files_dir"]
        print(f"[{datetime.now()}] Saving plots in main output directory")
    
    try:
        # 1. Velocity plot with uncertainty
        fig1 = plt.figure(figsize=(12, 8))
        plt.plot(vc_out["time_f"] * 1e6, vc_out["velocity_f_smooth"], 'b-', linewidth=2, label='Smoothed Velocity')
        plt.fill_between(vc_out["time_f"] * 1e6, 
                         vc_out["velocity_f_smooth"] - iua_out["vel_uncert"] * inputs["uncert_mult"],
                         vc_out["velocity_f_smooth"] + iua_out["vel_uncert"] * inputs["uncert_mult"],
                         alpha=0.3, color='blue', label=f'Uncertainty (multiplied by {inputs["uncert_mult"]})')
        plt.xlabel('Time (μs)', fontsize=14)
        plt.ylabel('Velocity (m/s)', fontsize=14)
        plt.title('Velocity vs Time with Uncertainty', fontsize=16)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"{base_filename}--velocity_with_uncertainty.png"), dpi=300, bbox_inches='tight')
        plt.close(fig1)
        
        # 2. IQ Analysis plot
        fig2 = plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.plot(sdf_out["time"] * 1e6, np.real(cf_out["voltage_filt"]), 'b-', label='Real Part')
        plt.plot(sdf_out["time"] * 1e6, np.imag(cf_out["voltage_filt"]), 'r-', label='Imaginary Part')
        plt.xlabel('Time (μs)', fontsize=12)
        plt.ylabel('Amplitude', fontsize=12)
        plt.title('IQ Signal Components', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 1, 2)
        iq_magnitude = np.sqrt(np.real(cf_out["voltage_filt"])**2 + np.imag(cf_out["voltage_filt"])**2)
        plt.plot(sdf_out["time"] * 1e6, iq_magnitude, 'g-', linewidth=2)
        plt.xlabel('Time (μs)', fontsize=12)
        plt.ylabel('Magnitude', fontsize=12)
        plt.title('IQ Signal Magnitude', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"{base_filename}--iq_analysis.png"), dpi=300, bbox_inches='tight')
        plt.close(fig2)
        
        # 3. Smoothed vs Raw Velocity comparison
        fig3 = plt.figure(figsize=(12, 8))
        plt.plot(vc_out["time_f"] * 1e6, vc_out["velocity_f"], 'r-', alpha=0.7, label='Raw Velocity', linewidth=1)
        plt.plot(vc_out["time_f"] * 1e6, vc_out["velocity_f_smooth"], 'b-', linewidth=2, label='Smoothed Velocity')
        plt.xlabel('Time (μs)', fontsize=14)
        plt.ylabel('Velocity (m/s)', fontsize=14)
        plt.title('Raw vs Smoothed Velocity Comparison', fontsize=16)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"{base_filename}--velocity_comparison.png"), dpi=300, bbox_inches='tight')
        plt.close(fig3)
        
        # 4. Noise analysis plot
        fig4 = plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.plot(vc_out["time_f"] * 1e6, iua_out["inst_noise"], 'r-', linewidth=2)
        plt.xlabel('Time (μs)', fontsize=12)
        plt.ylabel('Noise Fraction', fontsize=12)
        plt.title('Instantaneous Noise Analysis', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 1, 2)
        plt.hist(iua_out["inst_noise"], bins=50, alpha=0.7, color='blue', edgecolor='black')
        plt.xlabel('Noise Fraction', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Noise Distribution', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"{base_filename}--noise_analysis.png"), dpi=300, bbox_inches='tight')
        plt.close(fig4)
        
        print(f"[{datetime.now()}] Simplified plots created successfully")
        
        # Create original ALPSS spectrogram plots
        print(f"[{datetime.now()}] Creating original ALPSS spectrogram plots...")
        
        # 1. Original voltage data plot
        fig5 = plt.figure(figsize=(12, 8))
        plt.plot(sdf_out["time"] / 1e-9, sdf_out["voltage"] * 1e3, label="Original Signal", c="tab:blue")
        plt.plot(sdf_out["time"] / 1e-9, np.real(vc_out["voltage_filt"]) * 1e3, label="Filtered Signal", c="tab:orange")
        plt.axvspan(sdf_out["t_doi_start"] / 1e-9, sdf_out["t_doi_end"] / 1e-9, 
                    color="tab:red", alpha=0.35, label="ROI")
        plt.xlabel("Time (ns)")
        plt.ylabel("Voltage (mV)")
        plt.legend()
        plt.title("Voltage Data")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"{base_filename}--voltage_data.png"), dpi=300, bbox_inches='tight')
        plt.close(fig5)
        
        # 2. Imported spectrogram
        fig6 = plt.figure(figsize=(12, 8))
        plt.imshow(10 * np.log10(sdf_out["mag"] ** 2), aspect="auto", origin="lower", 
                   extent=[sdf_out["t"][0] / 1e-9, sdf_out["t"][-1] / 1e-9, 
                          sdf_out["f"][0] / 1e9, sdf_out["f"][-1] / 1e9], cmap=inputs["cmap"])
        plt.colorbar(label="Power (dBm)")
        plt.xlabel("Time (ns)")
        plt.ylabel("Frequency (GHz)")
        plt.title("Spectrogram Original Signal")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"{base_filename}--imported_spectrogram.png"), dpi=300, bbox_inches='tight')
        plt.close(fig6)
        
        # 3. ROI spectrogram
        fig7 = plt.figure(figsize=(12, 8))
        plt.imshow(10 * np.log10(sdf_out["mag"] ** 2), aspect="auto", origin="lower", 
                   extent=[sdf_out["t"][0] / 1e-9, sdf_out["t"][-1] / 1e-9, 
                          sdf_out["f"][0] / 1e9, sdf_out["f"][-1] / 1e9], cmap=inputs["cmap"])
        plt.colorbar(label="Power (dBm)")
        plt.axvline(sdf_out["t_start_corrected"] / 1e-9, ls="-", c="r", linewidth=2)
        plt.ylim([inputs["freq_min"] / 1e9, inputs["freq_max"] / 1e9])
        plt.xlim([sdf_out["t_doi_start"] / 1e-9, sdf_out["t_doi_end"] / 1e-9])
        plt.xlabel("Time (ns)")
        plt.ylabel("Frequency (GHz)")
        plt.title("Spectrogram ROI")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"{base_filename}--roi_spectrogram.png"), dpi=300, bbox_inches='tight')
        plt.close(fig7)
        
        # 4. Filtered spectrogram
        fig8 = plt.figure(figsize=(12, 8))
        plt.imshow(cf_out["power_filt"], aspect="auto", origin="lower", 
                   extent=[cf_out["t_filt"][0] / 1e-9, cf_out["t_filt"][-1] / 1e-9, 
                          cf_out["f_filt"][0] / 1e9, cf_out["f_filt"][-1] / 1e9], cmap=inputs["cmap"])
        plt.colorbar(label="Power (dBm)")
        plt.axvline(sdf_out["t_start_corrected"] / 1e-9, ls="-", c="r", linewidth=2)
        plt.ylim([inputs["freq_min"] / 1e9, inputs["freq_max"] / 1e9])
        plt.xlim([sdf_out["t_doi_start"] / 1e-9, sdf_out["t_doi_end"] / 1e-9])
        plt.xlabel("Time (ns)")
        plt.ylabel("Frequency (GHz)")
        plt.title("Filtered Spectrogram ROI")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"{base_filename}--filtered_roi_spectrogram.png"), dpi=300, bbox_inches='tight')
        plt.close(fig8)
        
        # 5. Thresholded spectrogram
        fig9 = plt.figure(figsize=(12, 8))
        plt.imshow(sdf_out["th3"], aspect="auto", origin="lower", 
                   extent=[sdf_out["t"][0] / 1e-9, sdf_out["t"][-1] / 1e-9, 
                          sdf_out["f_doi"][0] / 1e9, sdf_out["f_doi"][-1] / 1e9], cmap=inputs["cmap"])
        plt.axvline(sdf_out["t_start_corrected"] / 1e-9, ls="-", c="r", linewidth=2)
        plt.ylim([inputs["freq_min"] / 1e9, inputs["freq_max"] / 1e9])
        plt.xlim([sdf_out["t_doi_start"] / 1e-9, sdf_out["t_doi_end"] / 1e-9])
        plt.xlabel("Time (ns)")
        plt.ylabel("Frequency (GHz)")
        plt.title("Thresholded Spectrogram")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"{base_filename}--thresholded_spectrogram.png"), dpi=300, bbox_inches='tight')
        plt.close(fig9)
        
        # 6. Voltage ROI
        fig10 = plt.figure(figsize=(12, 8))
        plt.plot(sdf_out["time"] / 1e-9, np.real(vc_out["voltage_filt"]) * 1e3, label="Filtered Signal", c="tab:blue")
        plt.plot(vc_out["time_f"] / 1e-9, iua_out["env_max_interp"] * 1e3, label="Signal Envelope", c="tab:red")
        plt.plot(vc_out["time_f"] / 1e-9, iua_out["env_min_interp"] * 1e3, c="tab:red")
        plt.xlabel("Time (ns)")
        plt.ylabel("Voltage (mV)")
        plt.legend()
        plt.title("Voltage ROI")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"{base_filename}--voltage_roi.png"), dpi=300, bbox_inches='tight')
        plt.close(fig10)
        
        # 7. Velocity spectrogram overlay
        fig11 = plt.figure(figsize=(12, 8))
        plt.imshow(10 * np.log10(sdf_out["mag"] ** 2), aspect="auto", origin="lower", 
                   extent=[sdf_out["t"][0] / 1e-9, sdf_out["t"][-1] / 1e-9, 
                          sdf_out["f"][0] / 1e9, sdf_out["f"][-1] / 1e9], cmap=inputs["cmap"])
        plt.colorbar(label="Power (dBm)")
        plt.ylim([inputs["freq_min"] / 1e9, inputs["freq_max"] / 1e9])
        plt.xlim([sdf_out["t_doi_start"] / 1e-9, sdf_out["t_doi_end"] / 1e-9])
        plt.xlabel("Time (ns)")
        plt.ylabel("Frequency (GHz)")
        plt.title("Velocity Spectrogram Overlay")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"{base_filename}--velocity_spectrogram_overlay.png"), dpi=300, bbox_inches='tight')
        plt.close(fig11)
        
        # 8. Velocity uncertainty plot
        fig12 = plt.figure(figsize=(12, 8))
        plt.plot(vc_out["time_f"] * 1e6, iua_out["vel_uncert"], 'r-', linewidth=2)
        plt.xlabel('Time (μs)', fontsize=14)
        plt.ylabel('Velocity Uncertainty (m/s)', fontsize=14)
        plt.title('Velocity Uncertainty vs Time', fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"{base_filename}--velocity_uncertainty.png"), dpi=300, bbox_inches='tight')
        plt.close(fig12)
        
        # 9. Noise fraction plot
        fig13 = plt.figure(figsize=(12, 8))
        plt.plot(vc_out["time_f"] * 1e6, iua_out["inst_noise"], 'r-', linewidth=2)
        plt.xlabel('Time (μs)', fontsize=14)
        plt.ylabel('Noise Fraction', fontsize=14)
        plt.title('Noise Fraction vs Time', fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"{base_filename}--noise_fraction.png"), dpi=300, bbox_inches='tight')
        plt.close(fig13)
        
        # 10. Noise histogram
        fig14 = plt.figure(figsize=(12, 8))
        plt.hist(iua_out["noise"] * 1e3, bins=50, rwidth=0.8, alpha=0.7, color='blue', edgecolor='black')
        plt.xlabel('Noise (mV)', fontsize=14)
        plt.ylabel('Counts', fontsize=14)
        plt.title('Voltage Noise Distribution', fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"{base_filename}--noise_histogram.png"), dpi=300, bbox_inches='tight')
        plt.close(fig14)
        
        # 11. IQ amplitude plot
        fig15 = plt.figure(figsize=(12, 8))
        iq_amplitude = np.sqrt(np.real(cf_out["voltage_filt"])**2 + np.imag(cf_out["voltage_filt"])**2)
        plt.plot(sdf_out["time"] * 1e6, iq_amplitude, 'g-', linewidth=2)
        plt.xlabel('Time (μs)', fontsize=14)
        plt.ylabel('IQ Amplitude', fontsize=14)
        plt.title('IQ Signal Amplitude', fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"{base_filename}--IQ_amplitude.png"), dpi=300, bbox_inches='tight')
        plt.close(fig15)
        
        print(f"[{datetime.now()}] Original ALPSS spectrogram plots created successfully")
        return plt.figure()
    except Exception as e:
        print(f"[{datetime.now()}] ERROR in simple_plotting: {e}\n{traceback.format_exc()}")
        return None


# function for saving all the final outputs
def saving(
    sdf_out, cen, vc_out, sa_out, iua_out, fua_out, start_time, end_time, fig, **inputs
):
    print(f"[{datetime.now()}] Entered saving function")
    
    # Check if user wants to save all plots in subfolder
    save_all_plots = inputs.get("save_all_plots", "no")
    base_filename = inputs["filename"][0:-4]  # Remove file extension
    
    if save_all_plots == "yes":
        # Create subfolder for this file's plots
        plots_subfolder = os.path.join(inputs["out_files_dir"], f"{base_filename}_plots")
        os.makedirs(plots_subfolder, exist_ok=True)
        print(f"[{datetime.now()}] Saving main plots in subfolder: {plots_subfolder}")
        plot_dir = plots_subfolder
    else:
        # Save plots in main output directory
        plot_dir = inputs["out_files_dir"]
        print(f"[{datetime.now()}] Saving main plots in main output directory")
    
    try:
        # Save the main plots.png if figure exists and user wants plots
        if fig is not None and save_all_plots == "yes":
            print(f"[{datetime.now()}] Saving main plots.png...")
            fig.savefig(
                fname=os.path.join(plot_dir, base_filename + "--plots.png"),
                dpi="figure",
                format="png",
                facecolor="w",
            )
            print(f"[{datetime.now()}] Saved main plots.png.")
        else:
            print(f"[{datetime.now()}] Skipping main plots.png (fig is None or save_all_plots is 'no')")
        inputs_df = pd.DataFrame.from_dict(inputs, orient="index", columns=["Input"])
        print(f"[{datetime.now()}] Saving inputs_df...")
        inputs_df.to_csv(
            os.path.join(inputs["out_files_dir"], inputs["filename"][0:-4] + "--inputs" + ".csv"), index=True, header=False
        )
        print(f"[{datetime.now()}] Saved inputs_df.")
        velocity_data = np.stack((vc_out["time_f"], vc_out["velocity_f"]), axis=1)
        print(f"[{datetime.now()}] Saving velocity data...")
        np.savetxt(
            os.path.join(inputs["out_files_dir"], inputs["filename"][0:-4] + "--velocity" + ".csv"), velocity_data, delimiter=","
        )
        print(f"[{datetime.now()}] Saved velocity data.")
        velocity_data_smooth = np.stack(
            (vc_out["time_f"], vc_out["velocity_f_smooth"]), axis=1
        )
        print(f"[{datetime.now()}] Saving velocity smooth...")
        np.savetxt(
            os.path.join(inputs["out_files_dir"], inputs["filename"][0:-4] + "--velocity--smooth" + ".csv"),
            velocity_data_smooth,
            delimiter=",",
        )
        print(f"[{datetime.now()}] Saved velocity smooth.")
        voltage_data = np.stack(
            (
                sdf_out["time"],
                np.real(vc_out["voltage_filt"]),
                np.imag(vc_out["voltage_filt"]),
            ),
            axis=1,
        )
        print(f"[{datetime.now()}] Saving voltage data...")
        np.savetxt(
            os.path.join(inputs["out_files_dir"], inputs["filename"][0:-4] + "--voltage" + ".csv"), voltage_data, delimiter=","
        )
        print(f"[{datetime.now()}] Saved voltage data.")
        noise_data = np.stack((vc_out["time_f"], iua_out["inst_noise"]), axis=1)
        print(f"[{datetime.now()}] Saving noise data...")
        np.savetxt(
            os.path.join(inputs["out_files_dir"], inputs["filename"][0:-4] + "--noise--frac" + ".csv"), noise_data, delimiter=","
        )
        print(f"[{datetime.now()}] Saved noise data.")
        vel_uncert_data = np.stack((vc_out["time_f"], iua_out["vel_uncert"]), axis=1)
        print(f"[{datetime.now()}] Saving velocity uncertainty data...")
        np.savetxt(
            os.path.join(inputs["out_files_dir"], inputs["filename"][0:-4] + "--vel--uncert" + ".csv"),
            vel_uncert_data,
            delimiter=",",
        )
        print(f"[{datetime.now()}] Saved velocity uncertainty data.")
        results_to_save = {
            "Name": [
                "Date",
                "Time",
                "File Name",
                "Run Time",
                "Velocity at Max Compression",
                "Time at Max Compression",
                "Velocity at Max Tension",
                "Time at Max Tension",
                "Velocity at Recompression",
                "Time at Recompression",
                "Carrier Frequency",
                "Spall Strength",
                "Spall Strength Uncertainty",
                "Strain Rate",
                "Strain Rate Uncertainty",
                "Peak Shock Stress",
                "Spect Time Res",
                "Spect Freq Res",
                "Spect Velocity Res",
                "Signal Start Time",
                "Smoothing Characteristic Time",
            ],
            "Value": [
                start_time.strftime("%b %d %Y"),
                start_time.strftime("%I:%M %p"),
                inputs["filename"],
                (end_time - start_time),
                sa_out["v_max_comp"],
                sa_out["t_max_comp"],
                sa_out["v_max_ten"],
                sa_out["t_max_ten"],
                sa_out["v_rc"],
                sa_out["t_rc"],
                cen,
                sa_out["spall_strength_est"],
                fua_out["spall_uncert"],
                sa_out["strain_rate_est"],
                fua_out["strain_rate_uncert"],
                (0.5 * inputs["density"] * inputs["C0"] * sa_out["v_max_comp"]),
                sdf_out["t_res"],
                sdf_out["f_res"],
                0.5 * (inputs["lam"] * sdf_out["f_res"]),
                sdf_out["t_start_corrected"],
                iua_out["tau"],
            ],
        }
        print(f"[{datetime.now()}] Saving results_df...")
        results_df = pd.DataFrame(data=results_to_save)
        results_df.to_csv(
            os.path.join(inputs["out_files_dir"], inputs["filename"][0:-4] + "--results" + ".csv"), index=False, header=False
        )
        print(f"[{datetime.now()}] Saved results_df.")
        vel_smooth_with_uncert = np.stack(
            (
                vc_out["time_f"],
                vc_out["velocity_f_smooth"],
                iua_out["vel_uncert"],  # Uncertainty
                vc_out["velocity_f_smooth"] + iua_out["vel_uncert"],  # Velocity + Uncertainty
            ),
            axis=1,
        )
        print(f"[{datetime.now()}] Saving vel_smooth_with_uncert...")
        np.savetxt(
            os.path.join(inputs["out_files_dir"], inputs["filename"][0:-4] + "--vel-smooth-with-uncert" + ".csv"),
            vel_smooth_with_uncert,
            delimiter=",",
        )
        print(f"[{datetime.now()}] Saved vel_smooth_with_uncert.")
    except Exception as e:
        print(f"[{datetime.now()}] ERROR in saving: {e}\n{traceback.format_exc()}")


# function to pull out important points on the spall signal
def spall_analysis(vc_out, iua_out, **inputs):
    # if user wants to pull out the spall points
    if inputs["spall_calculation"] == "yes":

        # unpack dictionary values in to individual variables
        time_f = vc_out["time_f"]
        velocity_f_smooth = vc_out["velocity_f_smooth"]
        pb_neighbors = inputs["pb_neighbors"]
        pb_idx_correction = inputs["pb_idx_correction"]
        rc_neighbors = inputs["rc_neighbors"]
        rc_idx_correction = inputs["rc_idx_correction"]
        C0 = inputs["C0"]
        density = inputs["density"]
        freq_uncert = iua_out["freq_uncert"]
        vel_uncert = iua_out["vel_uncert"]

        # get the global peak velocity
        peak_velocity_idx = np.argmax(velocity_f_smooth)
        peak_velocity = velocity_f_smooth[peak_velocity_idx]

        # get the uncertainities associated with the peak velocity
        peak_velocity_freq_uncert = freq_uncert[peak_velocity_idx]
        peak_velocity_vel_uncert = vel_uncert[peak_velocity_idx]

        # attempt to get the fist local minimum after the peak velocity to get the pullback
        # velocity. 'order' is the number of points on each side to compare to.
        try:

            # get all the indices for relative minima in the domain, order them, and take the first one that occurs
            # after the peak velocity
            rel_min_idx = signal.argrelmin(velocity_f_smooth, order=pb_neighbors)[0]
            extrema_min = np.append(rel_min_idx, np.argmax(velocity_f_smooth))
            extrema_min.sort()
            max_ten_idx = extrema_min[
                np.where(extrema_min == np.argmax(velocity_f_smooth))[0][0]
                + 1
                + pb_idx_correction
            ]

            # get the uncertainities associated with the max tension velocity
            max_ten_freq_uncert = freq_uncert[max_ten_idx]
            max_ten_vel_uncert = vel_uncert[max_ten_idx]

            # get the velocity at max tension
            max_tension_velocity = velocity_f_smooth[max_ten_idx]

            # calculate the pullback velocity
            pullback_velocity = peak_velocity - max_tension_velocity

            # calculate the estimated strain rate and spall strength
            strain_rate_est = (
                (0.5 / C0)
                * pullback_velocity
                / (time_f[max_ten_idx] - time_f[np.argmax(velocity_f_smooth)])
            )
            spall_strength_est = 0.5 * density * C0 * pullback_velocity

            # set final variables for the function return
            t_max_comp = time_f[np.argmax(velocity_f_smooth)]
            t_max_ten = time_f[max_ten_idx]
            v_max_comp = peak_velocity
            v_max_ten = max_tension_velocity

        # if the program fails to find the peak and pullback velocities, then input nan's and continue with the program
        except Exception:
            print(traceback.format_exc())
            print("Could not locate the peak and/or pullback velocity")
            t_max_comp = np.nan
            t_max_ten = np.nan
            v_max_comp = np.nan
            v_max_ten = np.nan
            strain_rate_est = np.nan
            spall_strength_est = np.nan
            max_ten_freq_uncert = np.nan
            max_ten_vel_uncert = np.nan

        # try to get the recompression peak that occurs after pullback
        try:
            # get first local maximum after pullback
            rel_max_idx = signal.argrelmax(velocity_f_smooth, order=rc_neighbors)[0]
            extrema_max = np.append(rel_max_idx, np.argmax(velocity_f_smooth))
            extrema_max.sort()
            rc_idx = extrema_max[
                np.where(extrema_max == np.argmax(velocity_f_smooth))[0][0]
                + 2
                + rc_idx_correction
            ]
            t_rc = time_f[rc_idx]
            v_rc = velocity_f_smooth[rc_idx]

        # if finding the recompression peak fails then input nan's and continue
        except Exception:
            print(traceback.format_exc())
            print("Could not locate the recompression velocity")
            t_rc = np.nan
            v_rc = np.nan

    # if user does not want to pull out the spall points just set everything to nan
    else:
        t_max_comp = np.nan
        t_max_ten = np.nan
        t_rc = np.nan
        v_max_comp = np.nan
        v_max_ten = np.nan
        v_rc = np.nan
        spall_strength_est = np.nan
        strain_rate_est = np.nan
        peak_velocity_freq_uncert = np.nan
        peak_velocity_vel_uncert = np.nan
        max_ten_freq_uncert = np.nan
        max_ten_vel_uncert = np.nan

    # return a dictionary of the results
    sa_out = {
        "t_max_comp": t_max_comp,
        "t_max_ten": t_max_ten,
        "t_rc": t_rc,
        "v_max_comp": v_max_comp,
        "v_max_ten": v_max_ten,
        "v_rc": v_rc,
        "spall_strength_est": spall_strength_est,
        "strain_rate_est": strain_rate_est,
        "peak_velocity_freq_uncert": peak_velocity_freq_uncert,
        "peak_velocity_vel_uncert": peak_velocity_vel_uncert,
        "max_ten_freq_uncert": max_ten_freq_uncert,
        "max_ten_vel_uncert": max_ten_vel_uncert,
    }

    return sa_out


# function to find the specific domain of interest in the larger signal
def spall_doi_finder(**inputs):
    # import the desired data. Convert the time to skip and turn into number of rows
    t_step = 1 / inputs["sample_rate"]
    rows_to_skip = (
        inputs["header_lines"] + inputs["time_to_skip"] / t_step
    )  # skip the 5 header lines too
    nrows = inputs["time_to_take"] / t_step

    # change directory to where the data is stored
    os.chdir(inputs["exp_data_dir"])
    data = pd.read_csv(inputs["filename"], skiprows=int(rows_to_skip), nrows=int(nrows))

    # rename the columns of the data
    data.columns = ["Time", "Ampl"]

    # put the data into numpy arrays. Zero the time data
    time = data["Time"].to_numpy()
    time = time - time[0]
    voltage = data["Ampl"].to_numpy()

    # calculate the true sample rate from the experimental data
    fs = 1 / np.mean(np.diff(time))

    # calculate the short time fourier transform
    f, t, Zxx = stft(voltage, fs, **inputs)

    # Add IQ analysis section after loading time and voltage data
    def gaussian_window(M, std):
        n = np.arange(0, M) - (M - 1.0) / 2.0
        return np.exp(-0.5 * (n / std)**2)

    #   # Perform IQ analysis
    # Extract carrier frequency from input data
    N = len(voltage)
    fft_result = np.fft.fft(voltage)
    freq = np.fft.fftfreq(N, 1/fs)
    positive_freq_mask = freq > 0
    positive_freq = freq[positive_freq_mask]
    positive_fft = np.abs(fft_result[positive_freq_mask])

    # Find the frequency with maximum amplitude within the specified range
    freq_range_mask = (positive_freq >= inputs["freq_min"]) & (positive_freq <= inputs["freq_max"])
    carrier_idx = np.argmax(positive_fft[freq_range_mask])
    carrier_frequency = positive_freq[freq_range_mask][carrier_idx]

    print(f"Extracted carrier frequency: {carrier_frequency} Hz")
    
    # Demodulate signal
    I = voltage * np.cos(2 * np.pi * carrier_frequency * time)
    Q = voltage * np.sin(2 * np.pi * carrier_frequency * time)

    # Apply Gaussian smoothing with skip points
    skip_points = 100 # skipping initial points to avoid IQ analysis induced signal drop
    window_length = 801
    window = np.exp(-0.5 * (np.arange(0, window_length) - (window_length - 1.0) / 2.0) / 10**2)
    I_smooth = signal.convolve(I, window, mode='same')[skip_points:] / sum(window)
    Q_smooth = signal.convolve(Q, window, mode='same')[skip_points:] / sum(window)
    
   
    # Calculate amplitude and phase
    amplitude = np.sqrt(I_smooth**2 + Q_smooth**2)
    phase = np.unwrap(np.arctan2(Q_smooth, I_smooth))
 
    # Find initial stable amplitude
    initial_amplitude = np.mean(amplitude[:int(len(amplitude)/4.5)])
    # initial_phase = np.mean(phase[:int(len(phase)/4)])
    threshold = 0.4 * initial_amplitude
    
    # Detect start time using 50% amplitude drop
    start_index = np.where(amplitude < threshold)[0][0]
    # start_index = np.where(phase < threshold)[0][0]
    t_start_detected_iq = time[start_index]

    # Detect start time using multiple drops
    # start_index = 0
    # while start_index < len(amplitude):
    #     drop_indices = np.where(amplitude[start_index:] < threshold)[0]
    #     if len(drop_indices) == 0:
    #         break
        
    #     drop_start = start_index + drop_indices[0]
        
    #     # Check if signal rises again
    #     rise_indices = np.where(amplitude[drop_start:] > threshold)[0]
    #     if len(rise_indices) == 0:
    #         start_index = drop_start
    #         break
        
    #     rise_index = drop_start + rise_indices[0]
    #     start_index = rise_index

    # t_start_detected_iq = time[start_index]



    # After calculating amplitude, adjust time array to match
    time_adjusted = time[skip_points:skip_points+len(amplitude)]

    # Convert amplitude to mV and time to microseconds
    amplitude_mV = amplitude * 1e3
    time_us = time_adjusted * 1e6

    # Plot with matched array lengths and square aspect ratio
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
    ax1.plot(time_us, amplitude_mV, label='Complex Amplitude')
    ax1.plot(time_us, initial_amplitude * 1e3 * np.where(time_us < t_start_detected_iq * 1e6, 1, 0.5), 
            label='Step Function')
    ax1.axvline(x=t_start_detected_iq * 1e6, color='r', linestyle='--', 
                label='Start Time (IQ)')
    ax1.set_ylabel('Amplitude (mV)', fontsize=20)
    ax1.set_xlabel('Time (μs)', fontsize=20)
    ax1.legend(fontsize=12)
    ax1.tick_params(axis='both', labelsize=20)

    # Save IQ amplitude plot as a separate figure
    out_dir = inputs.get('out_files_dir', '.')
    fname_prefix = os.path.splitext(inputs.get('filename', 'ALPSS'))[0]
    fig_iq, ax_iq = plt.subplots(figsize=(8, 8))
    ax_iq.plot(time_us, amplitude_mV, label='Complex Amplitude')
    ax_iq.plot(time_us, initial_amplitude * 1e3 * np.where(time_us < t_start_detected_iq * 1e6, 1, 0.5), 
            label='Step Function')
    ax_iq.axvline(x=t_start_detected_iq * 1e6, color='r', linestyle='--', 
                label='Start Time (IQ)')
    ax_iq.set_ylabel('Amplitude (mV)', fontsize=20)
    ax_iq.set_xlabel('Time (μs)', fontsize=20)
    ax_iq.legend(fontsize=12)
    ax_iq.tick_params(axis='both', labelsize=20)
    plt.tight_layout()
    fig_iq.savefig(os.path.join(out_dir, f"{fname_prefix}--IQ_amplitude.png"), dpi=inputs.get('plot_dpi', 300), format='png', facecolor='w')
    plt.close(fig_iq)

    # Adjust phase plotting similarly
    ax2.plot(time_us, phase, label='Phase', color='green')
    ax2.set_xlabel('Time (μs)', fontsize=20)
    ax2.set_ylabel('Phase (radians)', fontsize=20)
    ax2.legend(fontsize=12)
    ax2.tick_params(axis='both', labelsize=20)
    plt.tight_layout()

    # Add IQ results to output dictionary
    sdf_out = {
        # ... (previous dictionary items)
        "amplitude_iq": amplitude,
        "phase_iq": phase,
        "t_start_detected_iq": t_start_detected_iq,
    }

    
## end of IQ analysis
 
    # calculate magnitude of Zxx
    mag = np.abs(Zxx)

    # calculate the time and frequency resolution of the transform
    t_res = np.mean(np.diff(t))
    f_res = np.mean(np.diff(f))

    # find the index of the minimum and maximum frequencies as specified in the user inputs
    freq_min_idx = np.argmin(np.abs(f - inputs["freq_min"]))
    freq_max_idx = np.argmin(np.abs(f - inputs["freq_max"]))

    # cut the magnitude and frequency arrays to smaller ranges
    mag_cut = mag[freq_min_idx:freq_max_idx, :]
    f_doi = f[freq_min_idx:freq_max_idx]

    # calculate spectrogram power
    power_cut = 10 * np.log10(mag_cut**2)


    # convert spectrogram powers to uint8 for image processing
    smin = np.min(power_cut)
    smax = np.max(power_cut)
    a = 255 / (smax - smin)
    b = 255 - a * smax
    power_gray = a * power_cut + b
    power_gray8 = power_gray.astype(np.uint8)

    # blur using a gaussian filter
    blur = cv.GaussianBlur(
        power_gray8, inputs["blur_kernel"], inputs["blur_sigx"], inputs["blur_sigy"]
    )

    # automated thresholding using Otsu's binarization
    ret3, th3 = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    # if not using a user input value for the signal start time
    if inputs["start_time_user"] == "none":

        # Find the position/row of the top of the binary spectrogram for each time/column
        col_len = th3.shape[1]  # number of columns
        row_len = th3.shape[0]  # number of columns
        top_line = np.zeros(col_len)  # allocate space to place the indices
        f_doi_top_line = np.zeros(
            col_len
        )  # allocate space to place the corresponding frequencies

        for col_idx in range(col_len):  # loop over every column
            for row_idx in range(row_len):  # loop over every row

                # moving from the top down, if the pixel is 255 then store the index and break to move to the next column
                idx_top = row_len - row_idx - 1

                if th3[idx_top, col_idx] == 255:
                    top_line[col_idx] = idx_top
                    f_doi_top_line[col_idx] = f_doi[idx_top]
                    break

        # if the signal completely drops out there will be elements of f_doi_top_line equal to zero - these points are
        # made NaNs. Same for top_line.
        f_doi_top_line_clean = f_doi_top_line.copy()
        f_doi_top_line_clean[np.where(top_line == 0)] = np.nan
        top_line_clean = top_line.copy()
        top_line_clean[np.where(top_line == 0)] = np.nan

        # find the index of t where the time is closest to the user input carrier_band_time
        carr_idx = np.argmin(np.abs(t - inputs["carrier_band_time"]))

        # calculate the average frequency of the top of the carrier band during carrier_band_time
        f_doi_carr_top_avg = np.mean(f_doi_top_line_clean[:carr_idx])

        # find the index in f_doi that is closest in frequency to f_doi_carr_top_avg
        f_doi_carr_top_idx = np.argmin(np.abs(f_doi - f_doi_carr_top_avg))

        # work backwards from the highest point on the signal top line until it matches or dips below f_doi_carr_top_idx
        highest_idx = np.argmax(f_doi_top_line_clean)
        for check_idx in range(highest_idx):
            cidx = highest_idx - check_idx - 1
            if top_line_clean[cidx] <= f_doi_carr_top_idx:
                break

        # add in the user correction for the start time
        t_start_detected_old = t[cidx]
        t_start_detected = t_start_detected_iq
        # t_start_detected = max(t_start_detected_iq, t_start_detected_old)
        # using mean of the IQ threshold drop and Jake's paper start detection (old) 
        # t_start_detected = np.mean([t_start_detected_iq, t_start_detected_old])

        t_start_corrected = t_start_detected + inputs["start_time_correction"]
        t_doi_start = t_start_corrected - inputs["t_before"]
        t_doi_end = t_start_corrected + inputs["t_after"]

        t_doi_start_spec_idx = np.argmin(np.abs(t - t_doi_start))
        t_doi_end_spec_idx = np.argmin(np.abs(t - t_doi_end))
        mag_doi = mag_cut[:, t_doi_start_spec_idx:t_doi_end_spec_idx]
        power_doi = 10 * np.log10(mag_doi**2)
        

    # if using a user input for the signal start time
    else:

        # these params become nan because they are only needed if the program
        # is finding the signal start time automatically
        f_doi_top_line_clean = np.nan
        carr_idx = np.nan
        f_doi_carr_top_idx = np.nan

        # use the user input signal start time to define the domain of interest
        t_start_detected = t[np.argmin(np.abs(t - inputs["start_time_user"]))]
        t_start_corrected = t_start_detected + inputs["start_time_correction"]
        t_doi_start = t_start_corrected - inputs["t_before"]
        t_doi_end = t_start_corrected + inputs["t_after"]

        t_doi_start_spec_idx = np.argmin(np.abs(t - t_doi_start))
        t_doi_end_spec_idx = np.argmin(np.abs(t - t_doi_end))
        mag_doi = mag_cut[:, t_doi_start_spec_idx:t_doi_end_spec_idx]
        power_doi = 10 * np.log10(mag_doi**2)
        

    cen=carrier_frequency # measured frequency 
    cen_idx = np.argmin(np.abs(f - cen))
    mag_cen = mag[cen_idx,:]
    fig, ax = plt.subplots(1,1)
    ax.plot(t, mag_cen)
    plt.tight_layout()
    plt.show()

    print(f"t_start_detected_old: {t_start_detected_old}")
    print(f"t_start_detected_iq: {t_start_detected_iq}")
    print(f"t_start_detected: {t_start_detected}")
    print(f"t_start_corrected: {t_start_corrected}")
  
    # dictionary to return outputs
    sdf_out = {
        "time": time,
        "voltage": voltage,
        "fs": fs,
        "f": f,
        "t": t,
        "Zxx": Zxx,
        "t_res": t_res,
        "f_res": f_res,
        "f_doi": f_doi,
        "mag": mag,
        "th3": th3,
        "f_doi_top_line_clean": f_doi_top_line_clean,
        "carr_idx": carr_idx,
        "f_doi_carr_top_idx": f_doi_carr_top_idx,
        "t_start_detected": t_start_detected,
        "t_start_corrected": t_start_corrected,
        "t_doi_start": t_doi_start,
        "t_doi_end": t_doi_end,
        "power_doi": power_doi,
    }

    return sdf_out

# function to calculate the short time fourier transform (stft) of a signal. ALPSS was originally built with a scipy
# STFT function that may now be deprecated in the future. This function seeks to roughly replicate the behavior of the
# legacy stft function, specifically how the time windows are calculated and how the boundaries are handled
def stft(voltage, fs, **inputs):
    if SHORTTIMEFFT_AVAILABLE:
        # Use the new scipy ShortTimeFFT class (scipy >= 1.9.0)
        SFT = ShortTimeFFT.from_window(
            inputs["window"],
            fs=fs,
            nperseg=inputs["nperseg"],
            noverlap=inputs["noverlap"],
            mfft=inputs["nfft"],
            scale_to="magnitude",
            phase_shift=None,
        )
        Sx_full = SFT.stft(voltage, padding="zeros")
        t_full = SFT.t(len(voltage))
        f = SFT.f

        # calculate the time array for the legacy scipy stft function without zero padding on the boundaries
        t_legacy = np.arange(
            inputs["nperseg"] / 2,
            voltage.shape[-1] - inputs["nperseg"] / 2 + 1,
            inputs["nperseg"] - inputs["noverlap"],
        ) / float(fs)

        # find the time index in the new stft function that corresponds to where the legacy function time array begins
        t_idx = np.argmin(np.abs(t_full - t_legacy[0]))

        # crop the time array to the length of the legacy function
        t_crop = t_full[t_idx : t_idx + len(t_legacy)]

        # crop the stft magnitude array to the length of the legacy function
        Sx_crop = Sx_full[:, t_idx : t_idx + len(t_legacy)]

        # return the frequency, time, and magnitude arrays
        return f, t_crop, Sx_crop
    else:
        # Fallback for older scipy versions - use the legacy stft function
        f, t, Zxx = signal.stft(
            voltage,
            fs=fs,
            window=inputs["window"],
            nperseg=inputs["nperseg"],
            noverlap=inputs["noverlap"],
            nfft=inputs["nfft"],
            return_onesided=True,
            boundary='zeros',
            padded=True
        )
        return f, t, np.abs(Zxx)

# function for smoothing the padded velocity data; padded data is used so the program can return
# a smooth velocity over the full domain of interest without running in to issues with the boundaries
def smoothing(
    velocity_pad,
    smoothing_window,
    smoothing_wid,
    smoothing_amp,
    smoothing_sigma,
    smoothing_mu,
):
    # if the smoothing window is not an odd integer exit the program
    if (smoothing_window % 2 != 1) or (smoothing_window >= len(velocity_pad) / 2):
        raise Exception(
            'Input variable "smoothing_window" must be an odd integer and less than half the length of '
            "the velocity signal"
        )

    # number of points to either side of the point of interest
    half_space = int(np.floor(smoothing_window / 2))

    # weights to be applied to each sliding window as calculated from a normal distribution
    weights = gauss(
        np.linspace(-smoothing_wid, smoothing_wid, smoothing_window),
        smoothing_amp,
        smoothing_sigma,
        smoothing_mu,
    )

    # iterate over the domain and calculate the gaussian weighted moving average
    velocity_f_smooth = np.zeros(len(velocity_pad) - smoothing_window + 1)
    for i in range(half_space, len(velocity_f_smooth) + half_space):
        vel_pad_win = velocity_pad[i - half_space : i + half_space + 1]
        velocity_f_smooth[i - half_space] = np.average(vel_pad_win, weights=weights)

    # return the smoothed velocity
    return velocity_f_smooth

# Function to calculate the velocity from the filtered voltage signal
def velocity_calculation(
    spall_doi_finder_outputs, cen, carrier_filter_outputs, **inputs
):
    # Unpack dictionary values into individual variables
    fs = spall_doi_finder_outputs["fs"]
    time = spall_doi_finder_outputs["time"]
    voltage_filt = carrier_filter_outputs["voltage_filt"]
    freq_min = inputs["freq_min"]
    freq_max = inputs["freq_max"]
    lam = inputs["lam"]
    t_doi_start = spall_doi_finder_outputs["t_doi_start"]
    t_doi_end = spall_doi_finder_outputs["t_doi_end"]

    # Isolate signal. Filter out frequencies outside the range of interest
    numpts = len(time)
    freq = fftshift(np.arange((-numpts / 2), (numpts / 2)) * fs / numpts)
    filt = (freq > freq_min) & (freq < freq_max)
    voltage_filt = ifft(fft(voltage_filt) * filt)

    # Get indices in the time array closest to the domain start and end times
    time_start_idx = np.argmin(np.abs(time - t_doi_start))
    time_end_idx = np.argmin(np.abs(time - t_doi_end))

    # Unwrap the phase angle of the filtered voltage signal
    phas = np.unwrap(np.angle(voltage_filt), axis=0)

    # Take the numerical derivative using the central difference method with a 9-point stencil
    dpdt, dpdt_pad = num_derivative(
        phas, inputs["smoothing_window"], time_start_idx, time_end_idx, fs
    )

    # Convert the derivative into velocity
    velocity_pad = (lam / 2) * (dpdt_pad - cen)
    velocity_f = (lam / 2) * (dpdt - cen)

    # Crop the time array
    time_f = time[time_start_idx:time_end_idx]

    # Smooth the padded velocity signal using a moving average with Gaussian weights
    velocity_f_smooth = smoothing(
        velocity_pad=velocity_pad,
        smoothing_window=inputs["smoothing_window"],
        smoothing_wid=inputs["smoothing_wid"],
        smoothing_amp=inputs["smoothing_amp"],
        smoothing_sigma=inputs["smoothing_sigma"],
        smoothing_mu=inputs["smoothing_mu"],
    )

    # Return a dictionary of the outputs
    vc_out = {
        "time_f": time_f,
        "velocity_f": velocity_f,
        "velocity_f_smooth": velocity_f_smooth,
        "phasD2_f": dpdt,
        "voltage_filt": voltage_filt,
        "time_start_idx": time_start_idx,
        "time_end_idx": time_end_idx,
    }

    return vc_out

# %%
