import h5py
import numpy as np
import scipy.signal as scignal

class preproc:

    """
    class preproc:
        contains functions for spike and trigger detection including some basic filtering
    """

    def spike_detect(filename, rec_type, ch_voltage, fs = 10000):

        """
            Read electrophysiology data from hdf5 file and detect spikes in the voltage signal

            :param filename: '/path/to/example.h5'
            :param rec_type: enum('intracell', 'extracell') patch mode
            :param ch_voltage: 'name' of the recording channel containing a voltage signal recorded in gap-free mode
            :param fs: sampling rate in Hz
            :return:
                voltage_trace: array (1,len(recording)) with rawdata trace
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

        return voltage_trace, spiketimes,

    def trigger_detect(filename, ch_trigger):

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


