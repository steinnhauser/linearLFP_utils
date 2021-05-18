import argparse
import elephant as ele
import numpy as np
import scipy.signal as scpsig
import scipy.stats as st

def argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("-p", "--scale-factor", type=float,
        help="Factor which scales the network population.", default=1)

    parser.add_argument("-i", "--n-idx", type=float,
        help="Factor which scales the outside stimulus of the network.", default=1)

    parser.add_argument("-n", "--run-name", type=str,
        help="Run name variable. Leave empty to rewrite runs.",
        default='default_js')

    parser.add_argument("-c", "--case", type=int,
        help="Case variable. Mainly for plotting function purposes.",
        default=1)

    # custom module for the stim_study input.
    parser.add_argument("-s", "--signal", type=str,
        help="Signal variable. Input 'E:E' to only see that population.",
        default="E:E")

    # Insert additional custom variable for saving multiple spike trains
    parser.add_argument("--save-spike-trains", action="store_true",
        help="Variable for saving the spike trains to the data directory.",
        default=False)

    # Additional argument to save lfp signals appended to analysis directory
    parser.add_argument("--save-lfp-signals", action="store_true",
        help="Variable for saving the lfp signals to the data directory.",
        default=False)

    parser.add_argument("--custom-e-weight", type=float,
        help="Custom way to set the excitatory weight.",
        default=0.002)

    parser.add_argument("--custom-i-weight", type=float,
        help="Custom way to set the inhibitory weight.",
        default=0.02)

    parser.add_argument("--custom-group-index", type=float,
        help="Custom name for the group in which the FRs and LFP is saved.",
        default=-1)

    args = parser.parse_args()
    return args


def lowpass_filter(input_signal, dt, lowpass_freq=300, highpass_freq = None):
    """ Applies a simple lowpass filter to a signal for a given dt.
    dt value is assumed to be in ms. Applies the filter for <300Hz """
    filt_dict_lfp = {
        'highpass_frequency': highpass_freq, # singular matrix at 6
        'lowpass_frequency': lowpass_freq, # Hz
        'order': 4,
        'filter_function': 'filtfilt',
        'fs': (1. / dt) * 1000,
        'axis': -1
    }

    return ele.signal_processing.butter(np.array(input_signal), **filt_dict_lfp)


def calculate_sync_from_FR(firing_rates, dt, numpoints_sims, tstop):
    """ Function which takes in a long list of lists including firing rates.
    These are transformed into delta spike trains, and a Gaussian convolution
    is applied to them to generate some spread. Finally, the correlation between
    each and every signal is taken and an average is calculated from the upper
    triangular portion of the matrix. """

    num_cells = len(firing_rates)
    frs = np.zeros(num_cells, dtype=object)

    # generate the gaussian kernel
    gaussian_sizes = int((5 * 10 + 1)/dt) # 5ms window
    gaussian_width = int(5/dt)
    gauss_kernel = scpsig.gaussian(gaussian_sizes, std=gaussian_width)

    # loop through each of the firing rates and transform the signals
    for n, frlist in enumerate(firing_rates):
        # calculate delta trains
        delta_fr_train, _ = np.histogram(frlist,
                bins=numpoints_sims,
                range=(0, tstop))

        # convolve to gaussians and save.
        frs[n] = np.convolve(delta_fr_train, gauss_kernel, 'same')

    """ Now calculate the SPCC value from the matrix. """
    spcc_mtx = []

    # use the multiprocessing tool for this loop.
    for i in range(num_cells):
        print(f"Calculating S matrix... {i}/{num_cells}", end="\r")
        if all(frs[i] == 0):
            pass # PCC undefined for no spikes
        else:
            for j in range(num_cells):
                if j<=i or all(frs[j] == 0):
                    pass # discard diagonal and lower-triangle.
                else:
                    spcc_mtx.append(st.pearsonr(
                        frs[i], frs[j]
                    )[0])
    print("")
    # extract a 1d array of the non-nan values:
    return np.mean(spcc_mtx), np.std(spcc_mtx)
