import h5py
import numpy as np
import scipy.signal as scignal
import scipy.ndimage as scimage

class preproc:

    """
    class preproc:
        contains functions for spike and trigger detection including some basic filtering
    """

    def spike_detect(self, filename, rec_type, ch_voltage, fs = 10000):

        """
            Read electrophysiology data from hdf5 file and detect spikes in the voltage signal

            :param filename: '/path/to/example.h5'
            :param rec_type: enum('intracell', 'extracell') patch mode
            :param ch_voltage: 'name' of the recording channel containing a voltage signal recorded in gap-free mode
            :param fs: sampling rate in Hz
            :return:
                voltage_trace: array (1,len(recording)) with rawdata trace
                rec_len: scalar length of the recording
                spiketimes: array (1,nSpikes) with spiketimes in sample points
            """

        f = h5py.File(filename, 'r')

        # get group keys

        grp_keylist = [key for key in f.keys()]
        print('File contained the groups', grp_keylist)

        # check for number of channels recorded

        ch_keylist = [key for key in f['channels'].keys()]
        print('Number of recording channels:', len(ch_keylist))

        # extract data from recording channels

        rawdata = {}

        for ch in range(0, len(ch_keylist)):
            name_ch = f['channels'][ch_keylist[ch]][0].astype(str)  # 'ch{}_data'.format(ch)
            ch_grp = f[name_ch]  # get each channel group into hdf5 grp object
            keylist = [key for key in ch_grp.keys()]  # get key within one group
            rawdata[name_ch] = ch_grp[keylist[1]]['data'][:]  # initialize as section_00

            for sec in range(2, len(keylist)):
                ch_sec_tmp = ch_grp[keylist[sec]]
                dset = ch_sec_tmp['data'][:]  # get array
                rawdata[name_ch] = np.append(rawdata[name_ch], dset)

        # filter signal

        voltage_trace = rawdata[ch_voltage]

        fs = 10000

        wn0 = 45 / fs
        wn1 = 55 / fs
        b, a = scignal.butter(2, [wn0, wn1], 'bandstop',
                              analog=False)  # second order, critical frequency, type, analog or digital
        voltage_trace = scignal.filtfilt(b, a, voltage_trace)

        if rec_type == 'extracell':

            # determine threshold
            sigma = np.median(np.abs(voltage_trace) / .6745)
            thr = 5 * sigma
            print('Threshold is -', thr, 'mV')

            # threshold signal
            tmp = np.array(voltage_trace)
            thr_boolean = [tmp > -thr]
            tmp[thr_boolean] = 0

            # detect spikes as threshold crossings
            tmp[tmp != 0] = 1
            tmp = tmp.astype(int)
            tmp2 = np.append(tmp[1:len(tmp)], np.array([0], int))
            dif = tmp2 - tmp

            spiketimes = np.where(dif == -1)[0]

        if rec_type == 'intracell':
            sigma = np.median(np.abs(voltage_trace + np.abs(min(voltage_trace)))) / .6745

            d_voltage = np.append(np.array(np.diff(voltage_trace)), np.array(0))
            d_sigma = np.median(np.abs(d_voltage + np.abs(min(d_voltage))) / (.6745))

            tmp = np.array(voltage_trace + np.abs(min(voltage_trace)))
            d_tmp = np.array(d_voltage + np.abs(min(d_voltage)))

            tmp[tmp < sigma] = 0
            d_tmp[d_tmp < d_sigma] = 0
            tmp[tmp > sigma] = 1
            d_tmp[d_tmp > d_sigma] = 1

            tmp = tmp.astype(int)
            d_tmp = d_tmp.astype(int)
            tmp2 = np.append(tmp[1:len(tmp)], np.array([0], int))
            d_tmp2 = np.append(d_tmp[1:len(d_tmp)], np.array([0], int))

            dif = tmp2 - tmp
            d_dif = d_tmp2 - d_tmp

            spiketimes = np.where(d_dif == -1)[0]

            # double check spiketimes

            for s in range(len(spiketimes)):
                if np.any(dif[spiketimes[s]:spiketimes[s] + int(.01 * fs)] == -1):
                    continue
                else:
                    spiketimes = np.delete(spiketimes, s)

        rec_len = len(voltage_trace)

        return voltage_trace, rec_len, spiketimes,

    def trigger_detect(self, filename, ch_trigger):

        """
        :param filename: '/path/to/exmplae/file.h5'
        :param ch_trigger: 'name' of recording channel containing trigger signal in filename

         :returns
            trigger_trace: array (1, len(rec)) raw trigger trace
            triggertimes: array (1, nTrigger) trigger times in sample points

        """

        f = h5py.File(filename, 'r')

        # get group keys

        grp_keylist = [key for key in f.keys()]
        print('File contained the groups', grp_keylist)

        # check for number of channels recorded

        ch_keylist = [key for key in f['channels'].keys()]
        print('Number of recording channels:', len(ch_keylist))

        # extract data from recording channels

        rawdata = {}

        for ch in range(0, len(ch_keylist)):
            name_ch = f['channels'][ch_keylist[ch]][0].astype(str)  # 'ch{}_data'.format(ch)
            ch_grp = f[name_ch]  # get each channel group into hdf5 grp object
            keylist = [key for key in ch_grp.keys()]  # get key within one group
            rawdata[name_ch] = ch_grp[keylist[1]]['data'][:]  # initialize as section_00

            for sec in range(2, len(keylist)):
                ch_sec_tmp = ch_grp[keylist[sec]]
                dset = ch_sec_tmp['data'][:]  # get array
                rawdata[name_ch] = np.append(rawdata[name_ch], dset)

        trigger_trace = rawdata[ch_trigger]

        tmp = np.array(trigger_trace)
        thr_boolean = [tmp < 1]
        tmp[thr_boolean] = 0
        tmp[tmp != 0] = 1
        tmp2 = np.append(tmp[1:len(tmp)], [0])
        dif = tmp - tmp2
        triggertimes = np.where(dif == -1)[0]

        return trigger_trace, triggertimes

class stimuli:

    """
    contains functions for analysis of the light-dependet responses to different stimulus classes such as binary noise or moving bars
    """

    def ste(self, spiketimes, triggertimes, rec_len, mseq, freq = 5, deltat=1000, fs = 10000):

        """
            Calculate the spike-triggered stimulus ensemble from the given spiketimes vector and noise m-sequence

            :param spiketimes: array (1,nSpikes) containing the spiketimes in sample points
            :param triggertimes: array (1,nTrigger) containing the triggertimes in sample points
            :param mseq: string 'name' with the name and path of the noise m-sequence
            :param deltat: int tau (in ms) is the time lag considered before each spike for calculating the spike-triggered stimulus ensemble
            :returns
                ste: array (nSpikes,(tau+100)/100, stimDim[0] * stimDim[1]) with the spike triggered stimulus for each spike i and time step tau at ste(i,tau,:)
                stimDim: list (1,3) with the stimulus parameters x,y,length of the m-seq
            """

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

        print('Number of triggers detected matches mseq length:', len(triggertimes) == stimDim[2])

        stimInd = np.zeros([rec_len,1]).astype(int) - 1
        for n in range(len(triggertimes) - 1):
            stimInd[triggertimes[n]:triggertimes[n + 1] - 1] += int(n + 1)
        stimInd[triggertimes[len(triggertimes) - 1]:triggertimes[len(triggertimes) - 1] + (fs/freq)] += int(len(triggertimes))

        delta = int(deltat * fs/1000)
        k = 100

        spiketimes = spiketimes[spiketimes > triggertimes[0] + delta]
        spiketimes = spiketimes[spiketimes < triggertimes[len(triggertimes) - 1] + (fs/freq)]
        nspikes = len(spiketimes)


        ste = np.zeros([nspikes, (delta + 1000) / k, stimDim[0] * stimDim[1]])
        for st in range(nspikes):

            # calculate ste with time lags in steps of 10 ms
            for t in range(-1000, delta, k):
                ste[st, int((t + 1000) / k), :] = np.array(Frames[stimInd[spiketimes[st] - t]])

        return ste, stimDim

    def sta(self, ste, stimDim):

        """
        filter the spike-triggered ensemble and calculate the linear receptive field by averaging.
        singular value decomp for first spatial and temporal filter component

        :param ste: array (nSpikes,(tau+100)/100, stimDim[0] * stimDim[1]) with the spike triggered stimulus for each spike i and time step tau at ste(i,tau,:)
        :returns

            :return sta: array ((tau+100)/100 , stimDim[0], stimDim[1]) with spike-triggered average reshaped in x-y-dimensions
            :return kernel: array (tau+100/100,1) with the time kernel of the center pixel with highest standard deviation and its surround pixels
            :return u: array (tau+100/100,tau+100/100) from (u,s,v) = svd(sta)
            :return s: array (tau+100/100,stimDim[0]*stimDim[1]) from (u,s,v) = svd(sta)
            :return v: array (stimDim[0]*stimDim[1],stimDim[0]*stimDim[1]) from (u,s,v) = svd(sta)
        """
        # average and smooth sta with gaussian filter, reshape smoothed rf

        sta_raw = np.mean(ste, 0)
        sta = scimage.filters.gaussian_filter(sta_raw.reshape(sta_raw.shape[0], stimDim[0], stimDim[1]), [.2, .7, .7])

        # calculate time kernel from center pixel
        sd_map = np.std(sta, 0)
        idx_center = np.where(sd_map == np.max(sd_map))
        kernel = sta[:,idx_center[0],idx_center[1]]

        try:

            (u,s,v) = np.linalg.svd(sta_raw)

        except Exception as e_svd:

            print(e_svd)
            u = np.zeros([sta_raw.shape[0], sta_raw.shape[0]])
            v = np.zeros([sta_raw.shape[1], sta_raw.shape[1]])

        if np.mean(u[:,0]) != np.mean(kernel):
            u = -1*u

        if np.mean(kernel) < 0:
            idx_rf = np.where(kernel == min(kernel))[0][0]
        else:
            idx_rf = np.where(kernel == max(kernel))[0][0]

        if np.mean(v[0,:]) != np.mean(sta_raw[idx_rf,:]):
            v = -1 * v

        return sta, kernel, u, s, v








