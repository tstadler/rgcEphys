import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.ndimage as scimage
import scipy.optimize as scoptimize
import scipy.stats as scstats
from sklearn.cross_validation import KFold

from rgcEphys import RgcEphys

class lnp_fit:


    def stim_conv(filename, ch_voltage, ch_trigger, rec_type, mseq, freq=5, deltat=1000, fs=10000):

        """
        Convolves the stimulus with the time-kernel obtained from sta and returns the convolved stimulus
        together with the instantaneous spike-triggered average

        :arg filename: str '/path/to/example.h5'
        :arg ch_voltage: str 'name' of the recording channel containing a voltage signal recorded in gap-free mode
        :arg ch_trigger: str 'name' of the recording channel containing a trigger signal recorded in gap-free mode
        :arg rec_type: enum('intracell', 'extracell') patch mode
        :arg mseq: str '/path/to/mseq'
        :arg freq scalar stimulation frequency in Hz
        :arg deltat scalar time lag in s for calculating the time kernel
        :arg fs: scalar sampling rate in Hz

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

    def lnp(self,filename, ch_voltage, ch_trigger, rec_type, mseq, ll_fun, jac, w0, k, freq=5, fs=10000):

        """
        Convolve the stimulus ensemble with the time kernel obtained from STA and fit the instantaneous RF of an LNP model with the given likelihood fun

        :arg filename: str '/path/to/example.h5'
        :arg ch_voltage: str 'name' of the recording channel containing a voltage signal recorded in gap-free mode
        :arg ch_trigger: str 'name' of the recording channel containing a trigger signal recorded in gap-free mode
        :arg rec_type: enum('intracell', 'extracell') patch mode
        :arg mseq: str '/path/to/mseq'
        :arg ll_fun function pointer to the negative log-likelihood function
        :arg jac boolen indicating whether fun returns gradient as well, grad has dimension (n,1)
        :arg w0 array (n,1) initial rf guess
        :arg k scalar k-fold cross-validation performed on the data set
        :arg freq scalar stimulation frequency in Hz
        :arg fs: scalar sampling rate in Hz


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

        LNP_dict = {}
        LNP_dict.clear()
        LNP_dict['nLL train'] = []
        LNP_dict['nLL test'] = []
        LNP_dict['w'] = []
        LNP_dict['pred correct'] = []
        LNP_dict['pearson r'] = []
        LNP_dict['R2'] = []
        LNP_dict['pred psth'] = []
        LNP_dict['true psth'] = []

        kf = KFold(T, n_folds=k)
        for train, test in kf:
            res = scoptimize.minimize(ll_fun, w0, args=(s[:, train], y[train]), jac=jac, method='TNC')
            print(res.message, 'neg log-liklhd: ', res.fun)

            LNP_dict['nLL train'].append(res.fun)
            LNP_dict['nLL test'].append(ll_fun(res.x, s[:, test], y[test])[0])
            LNP_dict['w'].append(res.x)

            y_test = np.zeros(len(test))
            for t in range(len(test)):
                r = np.exp(np.dot(res.x, s[:, test[t]]))
                y_test[t] = np.random.poisson(lam=r)

            LNP_dict['pred correct'].append((sum(y_test == y[test]) / len(test)))
            LNP_dict['pearson r'].append(scstats.pearsonr(y_test, y[test])[0])
            LNP_dict['R2'].append(
                1 - np.sum(np.square(y[test] - y_test)) / np.sum(np.square(y[test] - np.mean(y[test]))))
            LNP_dict['pred psth'].append(y_test * freq)
            LNP_dict['true psth'].append(y[test] * freq)
        LNP_df = pd.DataFrame(LNP_dict)

        return LNP_df

    def lnp_exp(self, spiketimes, triggertimes, s_conv, k, freq=5, fs=10000):

        """
        Fit the instantaneous RF of an LNP model with exponential non-linearity

        :arg spiketimes: array (1 x nSpikes)
        :arg triggertimes: array (1 x T)
        :arg s_conv: array (T x n)
        :arg k scalar k-fold cross-validation performed on the data set
        :arg freq scalar stimulation frequency in Hz
        :arg fs: scalar sampling rate in Hz

        """
        s = np.transpose(s_conv)  # make it a (n x T) array

        (n, T) = s.shape

        spiketimes = spiketimes[spiketimes > triggertimes[0]]
        spiketimes = spiketimes[spiketimes < triggertimes[len(triggertimes) - 1] + (fs / freq)]

        # bin spiketimes as stimulus frames



        y = np.histogram(spiketimes, bins=T,
                         range=[triggertimes[0], triggertimes[len(triggertimes) - 1] + (fs / freq)])[0]
        w0 = np.zeros(n)

        kf = KFold(T, n_folds=k)
        LNP_dict = {}
        LNP_dict.clear()
        LNP_dict['nLL train'] = []
        LNP_dict['nLL test'] = []
        LNP_dict['w'] = []
        LNP_dict['pred correct'] = []
        LNP_dict['pearson r'] = []
        LNP_dict['R2'] = []
        LNP_dict['pred psth'] = []
        LNP_dict['true psth'] = []

        for train, test in kf:
            res = scoptimize.minimize(self.nLL, w0, args=(s[:, train], y[train]), jac=True, method='TNC')
            print(res.message, 'neg log-liklhd: ', res.fun)

            LNP_dict['nLL train'].append(res.fun)
            LNP_dict['nLL test'].append(self.nLL(res.x, s[:, test], y[test])[0])
            LNP_dict['w'].append(res.x)

            y_test = np.zeros(len(test))
            for t in range(len(test)):
                r = np.exp(np.dot(res.x, s[:, test[t]]))
                y_test[t] = np.random.poisson(lam=r)

            LNP_dict['pred correct'].append((sum(y_test == y[test]) / len(test)))
            LNP_dict['pearson r'].append(scstats.pearsonr(y_test, y[test])[0])
            LNP_dict['R2'].append(
                1 - np.sum(np.square(y[test] - y_test)) / np.sum(np.square(y[test] - np.mean(y[test]))))
            LNP_dict['pred psth'].append(y_test * freq)
            LNP_dict['true psth'].append(y[test] * freq)

        LNP_df = pd.DataFrame(LNP_dict)

        return LNP_df

    def lnp_exp_ridge(self, spiketimes, triggertimes, s_conv, k, theta, freq=5, fs=10000):

        """
        Fit the instantaneous RF of an LNP model with exponential non-linearity and ridge regression on the filter componennts

        :arg spiketimes: array (1 x nSpikes)
        :arg triggertimes: array (1 x T)
        :arg s_conv: array (T x n)
        :arg k scalar k-fold cross-validation performed on the data set
        :arg theta list (1 x nTheta) with values for the regularization argeter that should be cross-validated
        :arg freq scalar stimulation frequency in Hz
        :arg fs: scalar sampling rate in Hz

        """

        s = np.transpose(s_conv)  # make it a (n x T) array
        (n, T) = s.shape

        spiketimes = spiketimes[spiketimes > triggertimes[0]]
        spiketimes = spiketimes[spiketimes < triggertimes[len(triggertimes) - 1] + (fs / freq)]




        # bin spiketimes as stimulus frames

        y = np.histogram(spiketimes, bins=T,
                         range=[triggertimes[0], triggertimes[len(triggertimes) - 1] + (fs / freq)])[0]
        w0 = np.zeros(n)

        kf = KFold(T, n_folds=k)

        theta_dict = {}
        theta_dict['theta'] = theta
        theta_dict['nLL train'] = []
        theta_dict['nLL test'] = []
        theta_dict['nLL mean test'] = []
        theta_dict['pred perform'] = []
        theta_dict['mean pred perform'] = []

        for t in theta:

            nLL_temp_train = []
            nLL_temp_test = []
            pred_test = []
            for train, test in kf:
                res = scoptimize.minimize(self.nLL_ridge, w0, args=(s[:, train], y[train], t), jac=True, method='TNC')
                print(res.message, 'neg log-liklhd: ', res.fun)
                nLL_temp_train.append(res.fun)
                nLL_temp_test.append(self.nLL(res.x, s[:, test], y[test])[0])

                y_test = np.zeros(len(test))
                for t in range(len(test)):
                    r = np.exp(np.dot(res.x, s[:, test[t]]))
                    y_test[t] = np.random.poisson(lam=r)

                pred_test.append((sum(y_test == y[test]) / len(test)))

            theta_dict['pred perform'].append(pred_test)
            theta_dict['mean pred perform'].append(np.mean(pred_test))
            theta_dict['nLL train'].append(nLL_temp_train)
            theta_dict['nLL test'].append(nLL_temp_test)
            theta_dict['nLL mean test'].append(np.mean(nLL_temp_test))

        theta_df = pd.DataFrame(theta_dict)

        try:
            theta_opt = theta[np.where(theta_df['nLL mean test'] == min(theta_df['nLL mean test']))[0]]
        except Exception as e1:
            theta_opt = theta[np.where(theta_df['nLL mean test'] == min(theta_df['nLL mean test'])[0])[0]]

        LNP_dict = {}
        LNP_dict.clear()
        LNP_dict['nLL train'] = []
        LNP_dict['nLL test'] = []
        LNP_dict['w'] = []
        LNP_dict['pred correct'] = []
        LNP_dict['pearson r'] = []
        LNP_dict['R2'] = []
        LNP_dict['pred psth'] = []
        LNP_dict['true psth'] = []

        for train, test in kf:
            res = scoptimize.minimize(self.nLL_ridge, w0, args=(s[:, train], y[train], theta_opt), jac=True,
                                          method='TNC')
            print(res.message, 'neg log-liklhd: ', res.fun)

            LNP_dict['nLL train'].append(res.fun)
            LNP_dict['nLL test'].append(self.nLL(res.x, s[:, test], y[test])[0])
            LNP_dict['w'].append(res.x)

            y_test = np.zeros(len(test))
            for t in range(len(test)):
                r = np.exp(np.dot(res.x, s[:, test[t]]))
                y_test[t] = np.random.poisson(lam=r)

            LNP_dict['pred correct'].append((sum(y_test == y[test]) / len(test)))
            LNP_dict['pearson r'].append(scstats.pearsonr(y_test, y[test])[0])
            LNP_dict['R2'].append(
                1 - np.sum(np.square(y[test] - y_test)) / np.sum(np.square(y[test] - np.mean(y[test]))))
            LNP_dict['pred psth'].append(y_test * freq)
            LNP_dict['true psth'].append(y[test] * freq)
        LNP_df = pd.DataFrame(LNP_dict)

        return LNP_df, theta_df, theta_opt

    def lnp_exp_lasso(self, spiketimes, triggertimes, s_conv, k, theta, freq=5, fs=10000):

        """
        Fit the instantaneous RF of an LNP model with exponential non-linearity and ridge regression on the filter componennts

        :arg spiketimes: array (1 x nSpikes)
        :arg triggertimes: array (1 x T)
        :arg s_conv: array (T x n)
        :arg k scalar k-fold cross-validation performed on the data set
        :arg theta list (1 x nTheta) with values for the regularization argeter that should be cross-validated
        :arg freq scalar stimulation frequency in Hz
        :arg fs: scalar sampling rate in Hz

        """

        s = np.transpose(s_conv)  # make it a (n x T) array
        (n,T) = s.shape

        spiketimes = spiketimes[spiketimes > triggertimes[0]]
        spiketimes = spiketimes[spiketimes < triggertimes[len(triggertimes) - 1] + (fs / freq)]

        # bin spiketimes as stimulus frames

        y = np.histogram(spiketimes, bins=T,
                         range=[triggertimes[0], triggertimes[len(triggertimes) - 1] + (fs / freq)])[0]
        w0 = np.zeros(n)

        kf = KFold(T, n_folds=k)

        theta_dict = {}
        theta_dict['theta'] = theta
        theta_dict['nLL train'] = []
        theta_dict['nLL test'] = []
        theta_dict['nLL mean test'] = []
        theta_dict['pred perform'] = []
        theta_dict['mean pred perform'] = []

        for t in theta:

            nLL_temp_train = []
            nLL_temp_test = []
            pred_test = []
            for train, test in kf:
                res = scoptimize.minimize(self.nLL_lasso, w0, args=(s[:, train], y[train], t), jac=True, method='TNC')
                print(res.message, 'neg log-liklhd: ', res.fun)
                nLL_temp_train.append(res.fun)
                nLL_temp_test.append(self.nLL(res.x, s[:, test], y[test])[0])

                y_test = np.zeros(len(test))
                for t in range(len(test)):
                    r = np.exp(np.dot(res.x, s[:, test[t]]))
                    y_test[t] = np.random.poisson(lam=r)

                pred_test.append((sum(y_test == y[test]) / len(test)))

            theta_dict['pred perform'].append(pred_test)
            theta_dict['mean pred perform'].append(np.mean(pred_test))
            theta_dict['nLL train'].append(nLL_temp_train)
            theta_dict['nLL test'].append(nLL_temp_test)
            theta_dict['nLL mean test'].append(np.mean(nLL_temp_test))

        theta_df = pd.DataFrame(theta_dict)
        try:
            theta_opt = theta[np.where(theta_df['nLL mean test'] == min(theta_df['nLL mean test']))[0]]
        except Exception as e1:
            theta_opt = theta[np.where(theta_df['nLL mean test'] == min(theta_df['nLL mean test'])[0])[0]]

        LNP_dict = {}
        LNP_dict.clear()
        LNP_dict['nLL train'] = []
        LNP_dict['nLL test'] = []
        LNP_dict['w'] = []
        LNP_dict['pred correct'] = []
        LNP_dict['pearson r'] = []
        LNP_dict['R2'] = []
        LNP_dict['pred psth'] = []
        LNP_dict['true psth'] = []

        for train, test in kf:
            res = scoptimize.minimize(self.nLL_lasso, w0, args=(s[:, train], y[train], theta_opt), jac=True,
                                      method='TNC')
            print(res.message, 'neg log-liklhd: ', res.fun)

            LNP_dict['nLL train'].append(res.fun)
            LNP_dict['nLL test'].append(self.nLL(res.x, s[:, test], y[test])[0])
            LNP_dict['w'].append(res.x)

            y_test = np.zeros(len(test))
            for t in range(len(test)):
                r = np.exp(np.dot(res.x, s[:, test[t]]))
                y_test[t] = np.random.poisson(lam=r)

            LNP_dict['pred correct'].append((sum(y_test == y[test]) / len(test)))
            LNP_dict['pearson r'].append(scstats.pearsonr(y_test, y[test])[0])
            LNP_dict['R2'].append(
                1 - np.sum(np.square(y[test] - y_test)) / np.sum(np.square(y[test] - np.mean(y[test]))))
            LNP_dict['pred psth'].append(y_test * freq)
            LNP_dict['true psth'].append(y[test] * freq)
        LNP_df = pd.DataFrame(LNP_dict)

        return LNP_df, theta_df, theta_opt

    def nLL(wT, s, y):
        """
        Compute the negative log-likelihood of an LNP model wih exponential non-linearity

        :arg wT: current receptive field array(stimDim[0]*stimDim[1],)
        :arg s: stimulus array(stimDim[0]*stimDim[1],T)
        :arg y: spiketimes array(,T)

        :return nLL: computed negative log-likelihood scalar
        :return dnLL: computed first derivative of the nLL
        """

        r = np.exp(np.dot(wT, s))
        nLL = np.dot(r - y * np.log(r), np.ones(y.shape))

        dnLL = np.dot(s * r - y * s, np.ones(y.shape))

        return nLL, dnLL

    def nLL_ridge(wT, s, y, theta):

        """
        Compute the negative log-likelihood of an LNP model wih exponential non-linearity with a L2 regularization

        :arg wT: current receptive field array(stimDim[0]*stimDim[1],)
        :arg s: stimulus array(stimDim[0]*stimDim[1],T)
        :arg y: spiketimes array(,T)
        :arg theta: list with values for the tuning parameter that should be tested

        :return nLL: computed negative log-likelihood scalar
        :return dnLL: computed first derivative of the nLL
        """

        r = np.exp(np.dot(wT, s))
        nLL = np.dot(r - y * np.log(r), np.ones(y.shape)) + theta * np.dot(wT, wT)

        dnLL = np.dot(s * r - y * s, np.ones(y.shape)) + 2 * theta * wT

        return nLL, dnLL

    def nLL_lasso(wT, s, y, theta):

        """
        Compute the negative log-likelihood of an LNP model wih exponential non-linearity

        :arg wT: current receptive field array(stimDim[0]*stimDim[1],)
        :arg s: stimulus array(stimDim[0]*stimDim[1],T)
        :arg y: spiketimes array(,T)
        :arg theta: list with values for the tuning parameter that should be tested

        :return nLL: computed negative log-likelihood scalar
        :return dnLL: computed first derivative of the nLL
        """

        r = np.exp(np.dot(wT, s))
        nLL = np.dot(r - y * np.log(r), np.ones(y.shape)) + theta * np.dot(np.abs(wT), np.ones(len(wT)))

        dnLL = np.dot(s * r - y * s, np.ones(y.shape)) + theta * np.ones(wT.shape)

        return nLL, dnLL

class plots:

    def cross_val_rf(lnp_df,stimDim,reg_type='no',theta_opt=0):

        """
        :arg lnp_df: pandas dataframe
        :arg reg_type: str 'type' of regularization
        """

        plt.rcParams.update({'figure.figsize': (15, 8)})

        fig_w_nLL, axarr = plt.subplots(2, int(lnp_df.shape[0] / 2))
        ax = axarr.reshape(lnp_df.shape[0])

        for ix, row in lnp_df.iterrows():
            im = ax[ix].imshow(row['w'].reshape(stimDim[0], stimDim[1]), interpolation='none', cmap=plt.cm.coolwarm)
            ax[ix].set_title('nLL: ' + str("%.2f" % round(row['nLL test'], 2)))
            ax[ix].set_xticklabels([])
            ax[ix].set_yticklabels([])

            fig_w_nLL.subplots_adjust(right=0.8)
            cbar_ax = fig_w_nLL.add_axes([0.85, 0.2, 0.02, 0.6])
            cbar = fig_w_nLL.colorbar(im, cax=cbar_ax)
            cbar.set_label('stimulus intensity', labelpad=40, rotation=270)
        plt.suptitle('Cross-validated RF with ' + reg_type +  ' regularization: ' + '$ \\theta $ = ' + str(theta_opt), size=20)

        return fig_w_nLL

    def psth_corr(lnp_df,freq=5):
        plt.rcParams.update({'figure.figsize': (15, 8), 'figure.subplot.wspace': .2, 'figure.subplot.hspace': .2,
                             'lines.linewidth': 1.5, 'axes.titlesize': 18})

        fig_corr, axarr = plt.subplots(2, int(lnp_df.shape[0] / 2), sharex=True, sharey=True)
        ax = axarr.reshape(lnp_df.shape[0])

        for ix, row in lnp_df.iterrows():
            t = np.linspace(0, len(row['pred psth']) / freq, len(row['pred psth']))

            plt.locator_params(axis='y', nbins=4)
            plt.locator_params(axis='x', nbins=4)

            ax[ix].plot(t, row['pred psth'])
            ax[ix].plot(t, row['true psth'])
            ax[ix].set_title('$\\rho$ : ' + str("%.2f" % round(row['pearson r'], 2)))
            ax[ix].set_xlim([0, len(row['pred psth']) / freq])

        fig_corr.tight_layout()
        fig_corr.text(0.5, -0.05, 'time [s]', fontsize=24, ha='center')
        fig_corr.text(-0.05, 0.5, 'PSTH', va='center', rotation='vertical', fontsize=24)

        return fig_corr

