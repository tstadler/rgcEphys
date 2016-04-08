import numpy as np
import scipy.ndimage as scimage
import scipy.optimize as scoptimize
import pandas as pd
from sklearn.cross_validation import KFold

import matplotlib.pyplot as plt
import seaborn as sns

import RgcEphys

class lnp_fit:


    def stim_conv(filename, ch_trigger, ch_voltage, rec_type, mseq, freq=5, deltat=1000, fs=10000):

        """
        Convolves the stimulus with the time-kernel obtained from sta and returns the convolved stimulus
        together with the instantaneous spike-triggered average

        """

        from IPython.display import display

        from matplotlib import ticker

        plt.rcParams.update({'figure.figsize': (10, 8), 'axes.titlesize': 20})

        (voltage_trace, rec_len, spiketimes) = RgcEphys.preproc.spike_detect(filename, rec_type, ch_voltage)

        (trigger_trace, triggertimes) = RgcEphys.preproc.trigger_detect(filename, ch_trigger)

        ste, stimDim = RgcEphys.stimuli.ste(spiketimes, triggertimes, rec_len, mseq=mseq, freq=freq, deltat=deltat,
                                            fs=fs)

        sta, kernel, u, s, v = RgcEphys.stimuli.sta(ste, stimDim)

        fig_rf_deltas = RgcEphys.plots.rf_deltas(sta)

        display(fig_rf_deltas)
        plt.close()

        fig_rf_svd = RgcEphys.plots.rf_svd(sta, kernel, u[:, 0], v[0, :])

        display(fig_rf_svd)
        plt.close()

        s_t_sep = bool(input('RF is space-time separable? (Yes: 1, No: 0)'))

        if s_t_sep:
            Frames = []
            stimDim = []

            lines = open(mseq + '.txt').read().split('\n')

            params = lines[0].split(',')
            stimDim.append(int(params[0]))
            stimDim.append(int(params[1]))
            stimDim.append(int(params[2]))

            nB = stimDim[0] * stimDim[1]

            for l in range(1, len(lines)):
                split = lines[l].split(',')
                Frame = np.array(split).astype(int)
                Frames.append(Frame)
            Frames = Frames - np.mean(Frames, 0)

            k = u[:, 0][::20]  # in time steps of 200 ms which corresponds to stimulation frequency of 5 Hz
            k_pad = np.vstack(
                (np.zeros(k[:, None].shape), k[:, None]))  # zero-padding to shift origin of weights vector accordingly

            F_conv = scimage.filters.convolve(Frames, k_pad)

            stimInd = np.zeros([rec_len, 1]).astype(int) - 1

            for n in range(len(triggertimes) - 1):
                stimInd[triggertimes[n]:triggertimes[n + 1] - 1] += int(n + 1)
            stimInd[triggertimes[len(triggertimes) - 1]:triggertimes[len(triggertimes) - 1] + (fs / freq)] += int(
                len(triggertimes))

            spiketimes = spiketimes[spiketimes > triggertimes[0]]
            spiketimes = spiketimes[spiketimes < triggertimes[len(triggertimes) - 1] + (fs / freq)]
            nspiketimes = len(spiketimes)

            ste = np.zeros([nspiketimes, stimDim[0] * stimDim[1]])

            for s in range(nspiketimes):
                ste[s, :] = F_conv[stimInd[spiketimes[s]], :]

            sta_inst = np.mean(ste, 0)

            fig_sta_inst = plt.subplots()

            im = plt.imshow(sta_inst.reshape(stimDim[0], stimDim[1]), cmap=plt.cm.coolwarm, interpolation='none')
            cbi = plt.colorbar(im)

            tick_locator = ticker.MaxNLocator(nbins=6)
            cbi.locator = tick_locator
            cbi.update_ticks()
            plt.title('Instantaneous STA')

            return F_conv, sta_inst

    def lnp_exp(self,filename, ch_trigger, ch_voltage, rec_type, mseq, fun, jac, w0, k, freq=5, fs=10000):

        """
        Fit the instantaneous RF of an LNP model with exponential non-linearity

        :param s_conv array (stimDim[0]*stimDim[1],T) stimulus projected onto time kernel
        :param spiketimes array (1,nSpikes)
        :param triggertimes array(1,T)
        :param fun function pointer to the negative log-likelihood function
        :param jac boolen indicating whether fun returns gradient as well, grad has dimension (n,1)
        :param w0 array (n,1) initial rf guess
        :param k scalar k-fold cross-validation performed on the data set

        """
        (voltage_trace, rec_len, spiketimes) = RgcEphys.preproc.spike_detect(filename, rec_type, ch_voltage)

        (trigger_trace, triggertimes) = RgcEphys.preproc.trigger_detect(filename, ch_trigger)

        s_conv, sta_inst = self.stim_conv(filename, ch_trigger, ch_voltage, rec_type, mseq)

        s = np.transpose(s_conv)  # make it a (n x T) array

        spiketimes = spiketimes[spiketimes > triggertimes[0]]
        spiketimes = spiketimes[spiketimes < triggertimes[len(triggertimes) - 1] + (fs / freq)]

        # bin spiketimes as stimulus frames
        T = s.shape[1]

        y = np.histogram(spiketimes, bins=T,
                         range=[triggertimes[0], triggertimes[len(triggertimes) - 1] + (fs / freq)])[0]


        kf = KFold(T, n_folds=k)
        LNP_dict = {}
        LNP_dict.clear()
        LNP_dict['nLL'] = []
        LNP_dict['w'] = []
        LNP_dict['pred perform'] = []
        LNP_dict['r2'] = []

        for train, test in kf:
            res = scoptimize.minimize(fun, w0, args=(s[:, train], y[train]), jac=jac)
            print(res.message, 'neg log-liklhd: ', res.fun)

            LNP_dict['nLL'].append(res.fun)
            LNP_dict['w'].append(res.x)

            y_test = np.zeros(len(test))
            for t in range(len(test)):
                r = np.exp(np.dot(res.x, s[:, test[t]]))
                y_test[t] = np.random.poisson(lam=r)

            LNP_dict['pred perform'].append((sum(y_test == y[test]) / len(test)))
            LNP_dict['r2'].append(np.var(y[test]) / np.var(y_test))
        LNP_df = pd.DataFrame(LNP_dict)

        return LNP_df

