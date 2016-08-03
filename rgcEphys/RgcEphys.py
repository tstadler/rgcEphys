import h5py
import numpy as np
import scipy.signal as scignal
import scipy.ndimage as scimage
import scipy.optimize as scoptimize
import matplotlib.pyplot as plt
import seaborn as sns
import fnmatch
import hashlib
from itertools import chain
import os
import pandas as pd
import re

from IPython.display import display

class parse:

    def fileScan(self,dataDirectory,fileTypes = ['abf','ini','h5','smh','smp']):
        parser = parse()
        fileLocation_pathlist = list(chain(*[parser.findFileType('*.' + suffix, dataDirectory) for suffix in fileTypes]))
        fileLocation_table = parser.locationToTable(fileLocation_pathlist)

        return fileLocation_table

    def findFileType(self,fileType,directory): # Type: '*.ini'
        # Find .ini files in all folders and subfolders of a specified directory
        fileLocation = [os.path.join(dirpath, f)
        for dirpath, dirnames, files in os.walk(directory)
        for f in fnmatch.filter(files,fileType)]
        return fileLocation

    def parseFileLocation(self,fileLocation,targetString='/'):
        # Extract useful information from directory path
        for fileType in list(fileLocation.keys()):
            backslash = [m.start() for m in re.finditer(targetString, entry)]
            headerFiles[entry].loc['surname'] = ['string',entry[backslash[-4]+1:backslash[-3]]]
            headerFiles[entry].loc['date'] = ['string',entry[backslash[-3]+1:backslash[-2]]]
            headerFiles[entry].loc['nExperiment'] = ['string',entry[backslash[-2]+1:backslash[-1]]]
        return headerFiles

    def readSHA1(self,fileLocation):
        # Find SHA-1 for file at file location
        BLOCKSIZE = 65536
        hasher = hashlib.sha1()
        with open(fileLocation, 'rb') as afile:
            buf = afile.read(BLOCKSIZE)
            while len(buf) > 0:
                hasher.update(buf)
                buf = afile.read(BLOCKSIZE)
        return hasher.hexdigest()

    def locationToTable(self,fileLocation_pathlist,targetString='/',topDirectory='Data/'):
        # Specification of dataframe in which to store information about files
        tableColumns = ['Surname', 'Date', 'Experiment', 'Subexpr', 'Filename', 'Filetype', 'Path', 'SHA1']
        fileLocation_table = pd.DataFrame(columns=tableColumns)

        for itx in range(len(fileLocation_pathlist)):
            path = fileLocation_pathlist[itx]
            # Find filepath within top directory, typically the 'Data/' folder
            subDirectory = path[path.find(topDirectory)+len(topDirectory):]

            # Find location of backslashes, which demarcate folders
            backslash = [m.start() for m in re.finditer(targetString, subDirectory)]

            # Extract folder and file names
            file = subDirectory[backslash[-1]+1:]
            fileName = file[:file.find('.')]
            fileType = file[file.find('.'):]
            Surname = subDirectory[:backslash[0]]
            Date = subDirectory[backslash[0]+1:backslash[1]]

            Experiment = np.nan
            Subexpr = np.nan
            if len(backslash) > 2:
                Experiment = subDirectory[backslash[1]+1:backslash[2]]
            if len(backslash) > 3:
                Subexpr = subDirectory[backslash[2]+1:backslash[3]]

            SHA1 = np.nan
            # SHA1 = readSHA1(path)

            fileEntry = [Surname,Date,Experiment,Subexpr,fileName,fileType,path,SHA1]
            fileLocation_table.loc[itx] = fileEntry

        return fileLocation_table

    def abfRead(self,path,filename,path_h5):
        import stfio
        """
        Input:
        * path: '/path/to/recording/'
        * filename: 'filename' without extension, has to be saved with ending .abf
        * path_h5: '/path/' where converted file is written to

        Output:
        file converted to hdf5 format, saved at path_h5
        """
        rec = stfio.read(path)

        print(rec.comment)
        print(rec.date)
        print(rec.datetime)
        print(rec.dt)
        print(rec.xunits)

        d = os.path.dirname(path_h5)
        if not os.path.exists(d):
            os.makedirs(d)

        filename_new = filename+'.h5'

        rec.write(path_h5+filename_new)

        print('hdf5 was written as '+filename_new+' at '+path_h5)

    def meta_data(ini):

        from configparser import ConfigParser

        info = {}

        config = ConfigParser()
        config.read(ini)
        info['zk'] = config.get('animal', 'zk')
        info['sex'] = config.get('animal', 'sex')
        info['birth'] = config.get('animal', 'birth')

        info['experimenter'] = config.get('experiment', 'experimenter')

        info['zoom'] = config.getfloat('morph', 'zoom')

        info['rec_type'] = config.get('ephys', 'rec_type')
        info['ch_voltage'] = config.get('ephys', 'ch_voltage')
        info['ch_trigger'] = config.get('ephys', 'ch_trigger')

        info['fname_noise'] = config.get('noise', 'filename')
        info['mseq'] = config.get('noise', 'm_seq')

        if '20Hz' in info['fname_noise']:
            info['freq'] = int(20)
        else:
            info['freq'] = int(5)

        if '20um' in info['fname_noise']:
            info['pixel_size'] = int(20)
        else:
            info['pixel_size'] = int(40)

        return info



class preproc:

    """
    class preproc:
        contains functions for spike and trigger detection including some basic filtering
    """

    def spike_detect(self, filename, rec_type, ch_voltage, fs = 10000):

        """
            Read electrophysiology data from hdf5 file and detect spiketimes in the voltage signal

            :param filename: '/path/to/example.h5'
            :param rec_type: enum('intracell', 'extracell') patch mode
            :param ch_voltage: 'name' of the recording channel containing a voltage signal recorded in gap-free mode
            :param fs: sampling rate in Hz
            :return:
                voltage_trace: array (1,len(recording)) with rawdata trace
                rec_len: scalar length of the recording
                spiketimes: array (1,nspiketimes) with spiketimes in sample points
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

            # detect spiketimes as threshold crossings
            tmp[tmp != 0] = 1
            tmp = tmp.astype(int)
            tmp2 = np.append(tmp[1:len(tmp)], np.array([0], int))
            dif = tmp2 - tmp

            spiketimes = np.where(dif == -1)[0]
            print('Number of spikes: ', len(spiketimes))

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
            print('Number of spikes: ', len(spiketimes))

        rec_len = len(voltage_trace)

        start = int(input('Plot voltage trace from (in s): '))
        end = int(input('to (in s): '))

        fig_v = plots.spiketimes(voltage_trace, spiketimes, start, end, fs = fs)

        display(fig_v)

        adjust0 = bool(int(input('Adjust threshold? (Yes: 1, No: 0): ')))
        plt.close(fig_v)
        if adjust0:
            if rec_type == 'extracell':
                adjust1 = True
                while adjust1:
                    pol = bool(int(input('y-axis switch? (Yes: 1, No: 0): ')))
                    alpha = int(input('Scale factor for threshold: '))

                    # determine threshold

                    thr = alpha * sigma

                    if pol:
                        print('Threshold is', thr, 'mV')
                        # threshold signal
                        tmp = np.array(voltage_trace)
                        thr_boolean = [tmp < thr]
                        tmp[thr_boolean] = 0

                        # detect spiketimes as threshold crossings
                        tmp[tmp != 0] = 1
                    else:
                        print('Threshold is -', thr, 'mV')
                        # threshold signal
                        tmp = np.array(voltage_trace)
                        thr_boolean = [tmp > -thr]
                        tmp[thr_boolean] = 0

                        tmp[tmp != 0] = 1

                    tmp = tmp.astype(int)
                    tmp2 = np.append(tmp[1:len(tmp)], np.array([0], int))
                    dif = tmp2 - tmp

                    spiketimes = np.where(dif == -1)[0]
                    print('Number of spikes: ', len(spiketimes))

                    fig_v = plots.spiketimes(voltage_trace, spiketimes, start, end, fs=fs)

                    display(fig_v)

                    adjust1 = bool(int(input('Adjust threshold again? (Yes: 1, No: 0): ')))
                    plt.close(fig_v)

        return voltage_trace, rec_len, spiketimes

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


class stimuli:

    """
    contains functions for analysis of the light-dependet responses to different stimulus classes such as binary noise or moving bars
    """

    def ste(spiketimes, triggertimes, rec_len, mseq, freq = 5, deltat=1000, fs = 10000):

        """
            Calculate the spike-triggered stimulus ensemble from the given spiketimes vector and noise m-sequence

            :param spiketimes: array (1,nspiketimes) containing the spiketimes in sample points
            :param triggertimes: array (1,nTrigger) containing the triggertimes in sample points
            :param mseq: string 'name' with the name and path of the noise m-sequence
            :param deltat: int tau (in ms) is the time lag considered before each spike for calculating the spike-triggered stimulus ensemble
            :returns
                ste: array (nspiketimes,(tau+100)/100, stimDim[0] * stimDim[1]) with the spike triggered stimulus for each spike i and time step tau at ste(i,tau,:)
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
        nspiketimes = len(spiketimes)


        ste = np.zeros([nspiketimes, (delta + 1000) / k, stimDim[0] * stimDim[1]])
        for st in range(nspiketimes):

            # calculate ste with time lags in steps of 10 ms
            for t in range(-1000, delta, k):
                ste[st, int((t + 1000) / k), :] = np.array(Frames[stimInd[spiketimes[st] - t]])

        return ste, stimDim

    def sta(ste, stimDim):

        """
        filter the spike-triggered ensemble and calculate the linear receptive field by averaging.
        singular value decomp for first spatial and temporal filter component

        :param ste: array (nspiketimes,(tau+100)/100, stimDim[0] * stimDim[1]) with the spike triggered stimulus for each spike i and time step tau at ste(i,tau,:)
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

        if np.sign(np.mean(u[:,0])) != np.sign(np.mean(kernel)):
            u = -1*u

        if np.mean(kernel) < 0:
            idx_rf = np.where(kernel == min(kernel))[0][0]
        else:
            idx_rf = np.where(kernel == max(kernel))[0][0]

        if np.sign(np.mean(v[0,:])) != np.sign(np.mean(sta_raw[idx_rf,:])):
            v = -1 * v

        return sta, kernel, u, s, v

    def chirp(spiketimes, triggertimes, delT = .1, fs=10000):

        """
        :param spiketimes: array (1, nspiketimes)
        :param triggertimes: array (1, nTrigger)
        :param fs: sampling rate of the recording
        :param delT: scalar binsize for the psth in s
        :return: 
                psth_trial:
                    array (ntrials,T/delT) with spike counts per trial binned with delT
                psth:
                    array (1,T/delT) where T is the length of one stimulus trial in s and psth is averaged over trials
                f_norm:
                    list [ntrials] [nspiketimes] list of ntrials arrays with the spiketimes of this trial
                    relative to trigger onset (0 = triggertimes[trial,0])
                loop_duration_s:
                    scalar time of one stimulus trial with imprecise stimulation frequency
        """

        # loop over trials and extract spike times per loop

        StimDuration = 32.5

        ntrials = int(np.floor(len(triggertimes)/2))

        triggertimes = triggertimes.reshape(ntrials, 2)

        true_loop_duration = []
        for trial in range(1, ntrials):
            true_loop_duration.append(triggertimes[trial, 0] - triggertimes[trial - 1, 0])

        loop_duration_n = np.ceil(np.mean(true_loop_duration))  # in sample points
        loop_duration_s = loop_duration_n / fs  # in s

        print('Due to imprecise stimulation freqeuncy a delta of', loop_duration_s - StimDuration,
              's was detected')

        f = []
        for trial in range(ntrials - 1):
            f.append(np.array(spiketimes[(spiketimes > triggertimes[trial, 0]) & (spiketimes < triggertimes[trial + 1, 0])]))
        f.append(np.array(
            spiketimes[(spiketimes > triggertimes[ntrials - 1, 0]) & (spiketimes < triggertimes[ntrials - 1, 0] + loop_duration_n)]))

        f_norm = []
        for trial in range(ntrials):
            f_norm.append(f[trial] - triggertimes[trial, 0])

        T = int(loop_duration_s)  # in s

        nbins1 = T / delT

        psth = np.zeros(nbins1)  # .astype(int)
        psth_trials = []

        for trial in range(ntrials):
            psth_trials.append(np.histogram(f_norm[trial] / fs, nbins1, [0, T])[0])
            psth += psth_trials[trial]

            psth = psth / (delT * ntrials)

        return (psth_trials, psth, f_norm, loop_duration_s)

    def ds(spiketimes, triggertimes):

        """
        :param spiketimes array (1, nSpikes)
        :param triggertimes array(1, nTrigger)
        :returns:
            :return spiketimes_trial list [ntrials] [nSpikes] list of ntrials arays containing spiketimes sorted by trial
            :return spiketimes_normed list [ntrials] [nSpikes] list of ntrials arrays with the spike train of this trial
            :return hist list [ntrials] [nconditions] list of ntrials array with the number of spikes per direction in this trial
            :return hist_sorted list [ntrial] [nconditions] list of ntrials arrays with the number of spikes per direction but sorted from 0 deg to 315 deg
            :return dsi scalar direction-selectivity index which is R_p-R_n/(R_p + R_n)
            :return deg array (1,nconditions) containing the tested directions ordered as they were presented
        """

        # fetch data

        nconditions = int(8)
        ntrials = int(np.floor(len(triggertimes) / 8))

        deg = np.array([0, 180, 45, 225, 90, 270, 135, 315])  # np.arange(0, 360, 360/nconditions)
        idx = np.array([0, 4, 6, 2, 5, 1, 7, 3])

        true_loop_duration = []
        for trial in range(1, ntrials):
            true_loop_duration.append(triggertimes[trial * nconditions] - triggertimes[(trial - 1) * nconditions])
        loop_duration_n = np.ceil(np.mean(true_loop_duration))  # in sample points
        # loopDuration_s = loopDuration_n/10000 # in s


        spiketimes_trial = []
        spiketimes_normed = []
        hist = []
        hist_sorted = []

        for trial in range(ntrials - 1):
            spiketimes_trial.append(np.array(
                spiketimes[(spiketimes > triggertimes[trial * nconditions]) & (spiketimes < triggertimes[(trial + 1) * nconditions])]))
            spiketimes_normed.append(spiketimes_trial[trial] - triggertimes[trial * nconditions])
            hist.append(np.histogram(spiketimes_normed[trial], 8, [0, loop_duration_n])[0])

            # sort by condition
            hist_sorted.append(hist[trial][idx])

        spiketimes_trial.append(np.array(spiketimes[(spiketimes > triggertimes[(ntrials - 1) * nconditions])
                                            & (spiketimes < triggertimes[(ntrials - 1) * nconditions] + loop_duration_n)]))
        spiketimes_normed.append(spiketimes_trial[ntrials - 1] - triggertimes[(ntrials - 1) * nconditions])
        hist.append(np.histogram(spiketimes_normed[ntrials - 1], 8, [0, loop_duration_n])[0])
        hist_sorted.append(hist[ntrials - 1][idx])

        hist_sum = np.sum(hist, 0)
        r_p = np.max(hist_sum)
        idx_p = np.where(hist_sum == r_p)[0][0]
        d_p = deg[idx_p]
        if (idx_p % 2) == 0:
            d_n = deg[idx_p + 1]
            r_n = hist_sum[idx_p + 1]
        else:
            d_n = deg[idx_p - 1]
            r_n = hist_sum[idx_p - 1]
        dsi = (r_p - r_n) / (r_p + r_n)

        return (spiketimes_trial, spiketimes_normed, hist, hist_sorted, dsi, deg)

class morph:


    """
    contains functions for analysing the morphologies of cells filled and reconstructed offline
    """

    def overlay(self, data_folder, exp_date, eye, cell_id, write_path, zoom, rf, pixel_size):
        """

        :param data_folder: str containing '/abs/path/to/data/
        :param exp_date: str with 'yyyy-mm-dd'
        :param eye: str enum('L','R')
        :param cell_id: int cell_id
        :param write_path: str containg '/abs/path/to/datawrite/
        :param zoom: double zoom factor used for recording morph
        :param rf: array (stimDim[0] x stimDim[1]) with the spike-triggered average or some other filter that should be overlayed with morph
        :param pixel_size: tuple (dy, dx) noise pixel side length in um
        :returns:
            :return rf_center array
            :return rf_up array
            :return morph array
            :return morph_pad array
        """

        morph = self.morph.import_stack(self, data_folder, exp_date, eye, cell_id, write_path)

        dx = pixel_size
        dy = pixel_size
        morph_size =  .64/zoom * 110  # side length of stack image in um
        scan_x = morph.shape[1]
        scan_y = morph.shape[0]

        print('morph side length: ', morph_size)
        print('with scanning mode: ', scan_x, 'x', scan_y)

        factor = (np.ceil(morph_size / dy), np.ceil(morph_size / dx))
        rf_even = scimage.zoom(rf, factor, order=0)

        center0 = rf_even.shape[0] / 2
        center1 = rf_even.shape[1] / 2

        nx = int(np.ceil(morph_size / dx)) * factor[1]  # number of pixels in x dimension covered by morph
        ny = int(np.ceil(morph_size / dy)) * factor[0]  # number of pixels in y dimensions covered by morph

        # cut out the region of noise covered by morph

        rf_center = rf_even[center0 - ny / 2:center0 + ny / 2, center1 - nx / 2:center1 + nx / 2]

        factor = (scan_y / rf_center.shape[0], scan_x / rf_center.shape[1])
        # print('upsample rf by a factor of:' , factor, 'with nearest neighbour interpolation')

        rf_center = scimage.zoom(rf_center, factor, order=0)

        # padding

        dx_morph = morph_size / scan_x  # morph pixel side length in um
        dy_morph = morph_size / scan_y  # morph pixel side length in um

        dely = (rf.shape[0] * dy - morph_size) / 2  # missing at each side of stack to fill stimulus in um
        delx = (rf.shape[1] * dx - morph_size) / 2

        ny_pad = int(dely / dy_morph)  # number of pixels needed to fill the gap
        nx_pad = int(delx / dx_morph)

        morph_pad = np.lib.pad(morph, ((ny_pad, ny_pad), (nx_pad, nx_pad)), 'constant', constant_values=0)

        factor = (morph_pad.shape[0] / rf.shape[0], morph_pad.shape[1] / rf.shape[1])
        # print('resampled by a factor of:' , factor, 'with nearest neighbour interpolation')

        rf_up = scimage.zoom(rf, factor, order=0)

        return rf_center, rf_up, morph, morph_pad

    def import_stack(self, data_folder, exp_date, eye, cell_id, write_path):
        import tifffile as tf

        # read from .tif

        full_path = data_folder + exp_date + '/' + eye + '/' + str(cell_id) + '/linestack.tif'

        stack = tf.imread(full_path)

        # smooth with a gaussian filter

        stack_smooth = scimage.filters.gaussian_filter(stack, [.2, .9, .9])

        # average along z-axis to get flattened top view

        morph = np.mean(stack_smooth, 0)
        morph = morph[::-1]

        # display morph

        plt.rcParams.update({
            'figure.figsize': (10, 8),
            'axes.titlesize': 20
        })

        fig = plt.figure()
        fig.suptitle(full_path)
        clim = (np.min(morph), np.max(morph) * .2)
        plt.imshow(morph, clim=clim)
        display(fig)
        plt.close(fig)

        # save array to file
        sv_path = write_path + exp_date + '/' + eye + '/' + str(cell_id) + '/morph'

        np.save(sv_path, morph)

        return morph

    def shift(self,morph, morph_pad, rf, rf_up, pixel_size, zoom):

        # Fit 2d Gauss to find soma in morph and rf center

        params_m = self.helper.fitgaussian(self,morph_pad)
        params_rf = self.helper.fitgaussian(self,np.abs(rf_up))

        (shift_y, shift_x) = (params_rf - params_m)[1:3]

        dx = pixel_size
        dy = pixel_size

        morph_size = .64 / zoom * 110  # side length of stack image in um
        scan_x = morph.shape[1]
        scan_y = morph.shape[0]

        dx_morph = morph_size / scan_x  # morph pixel side length in um
        dy_morph = morph_size / scan_y  # morph pixel side length in um

        dely = (rf.shape[0] * dy - morph_size) / 2  # missing at each side of stack to fill stimulus in um
        delx = (rf.shape[1] * dx - morph_size) / 2

        ny_pad = int(dely / dy_morph)  # number of pixels needed to fill the gap
        nx_pad = int(delx / dx_morph)

        morph_shift = np.lib.pad(morph, (
        (ny_pad + int(shift_y), ny_pad - int(shift_y)), (nx_pad + int(shift_x), nx_pad - int(shift_x))), 'constant',
                                     constant_values=0)
        params_m_shift = self.helper.fitgaussian(self, morph_shift)

        return morph_shift, params_m, params_m_shift, params_rf

class plots:

    """
    contains functions for plotting of analysis results
    """

    def rawtrace(voltage_trace, start, end, fs=10000):

        """
        :param voltage_trace array (1, rec_len) with the filtered raw trace
        :param start scalar start of plottted segment in s
        :param end scalar end of plottted segment in s
        :param fs scalar sampling rate in Hz
        """

        plt.rcParams.update(
            {'figure.figsize': (15, 6), 'axes.titlesize': 20, 'axes.labelsize': 18, 'xtick.labelsize': 16,
             'ytick.labelsize': 16})

        x = np.linspace(start, end, (end - start) * fs)

        fig, ax = plt.subplots()

        ax.plot(x, voltage_trace[start * fs:end * fs], linewidth=2)
        ax.set_ylabel('Voltage [mV]', labelpad=20)
        ax.set_xlabel('Time [s]', labelpad=20)
        ax.set_xlim([start, end])
        plt.locator_params(axis='y', nbins=5)

        return fig

    def spiketimes(voltage_trace,spiketimes, start, end, fs=10000):

        """
            :param voltage_trace array (1, rec_len) with the filtered raw trace
            :param spiketimes array (1,nSpikes) with spike times in sample points
            :param start scalar start of plottted segment in s
            :param end scalar end of plottted segment in s
            :param fs scalar sampling rate in Hz
            """

        plt.rcParams.update(
            {'figure.figsize': (15, 6), 'axes.titlesize': 20, 'axes.labelsize': 18, 'xtick.labelsize': 16,
             'ytick.labelsize': 16})

        fig, ax = plt.subplots()

        x = np.linspace(start, end, (end - start) * fs)
        n = spiketimes[(spiketimes > start * fs) & (spiketimes < end * fs)].astype(int)

        ax.plot(x, voltage_trace[start * fs:end * fs], linewidth=2)
        ax.plot(x[n - start * fs], voltage_trace[n], 'or')
        ax.set_xlim([start, end])
        ax.set_ylabel('Voltage [mV]', labelpad=20)
        ax.set_xlabel('Time [s]', labelpad=20)
        plt.locator_params(axis='y', nbins=5)

        return fig

    def rf_deltas(sta):

        """
        :param sta array(tau,stimDim[0],stimDim[1]) smoothed sta returned by stimuli.sta()
        """
        from matplotlib import ticker

        plt.rcParams.update({
            'figure.figsize': (15, 8), 'figure.subplot.hspace': 0, 'figure.subplot.wspace': .2, 'axes.titlesize': 16,
            'axes.labelsize': 18, 'xtick.labelsize': 16, 'ytick.labelsize': 16})

        sta_norm = sta / np.std(sta, 0)

        fig = plt.figure()
        tmp = 1

        if sta.shape[0] % 20 != 0:
            end = sta.shape[0] - sta.shape[0] % 20
        else:
            end = sta.shape[0]

        for delt in range(0, end, 10):
            fig.add_subplot(2, sta.shape[0] / 20, tmp)
            im = plt.imshow(sta_norm[delt, :, :],
                            cmap=plt.cm.coolwarm, clim=(-np.percentile(sta_norm, 95), np.percentile(sta_norm, 95)),
                            interpolation='none')
            plt.title('$\Delta$ t = ' + str(-(delt - 10) * 10) + 'ms')
            plt.yticks([])
            plt.xticks([])
            tmp += 1

        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.2, 0.02, 0.6])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label('s.d. units', labelpad=40, rotation=270)

        tick_locator = ticker.MaxNLocator(nbins=6)
        cbar.locator = tick_locator
        cbar.update_ticks()

        return fig

    def rf_contour(sta):

        """
        :param sta array(tau,stimDim[0],stimDim[1]) smoothed sta returned by stimuli.sta()
        """
        from matplotlib import ticker

        plt.rcParams.update({
            'figure.figsize': (10, 8), 'figure.subplot.hspace': .2, 'figure.subplot.wspace': .2, 'axes.titlesize': 16,
            'axes.labelsize': 18,
            'xtick.labelsize': 16, 'ytick.labelsize': 16, 'lines.linewidth': 4})

        tau = int(input('Select best time lag tau for rf mapping [in ms]: '))
        frame = int(10 - tau / 10)
        x1 = int(input('And the pixel borders: x1: '))
        x2 = int(input('And the pixel borders: x2: '))
        y1 = int(input('And the pixel borders: y1: '))
        y2 = int(input('And the pixel borders: y2: '))

        fig = plt.figure()

        im = plt.imshow(sta[frame, :, :][x1:x2, y1:y2], interpolation='none',
                        cmap=plt.cm.Greys_r, extent=(y1, y2, x2, x1), origin='upper')
        cs = plt.contour(sta[frame, :, :][x1:x2, y1:y2],
                         extent=(y1, y2, x2, x1), cmap=plt.cm.coolwarm, origin='upper', linewidth=4)

        cb = plt.colorbar(cs, extend='both', shrink=.8)
        cbaxes = fig.add_axes([.15, .02, .6, .03])  # [left, bottom, width, height]
        cbi = plt.colorbar(im, orientation='horizontal', cax=cbaxes)

        tick_locator = ticker.MaxNLocator(nbins=6)
        cbi.locator = tick_locator
        cbi.update_ticks()

        return fig

    def rf_svd(sta, kernel, u, v):

        """
        :param sta
        :param kernel
        :param u
        :param v
        """
        from matplotlib import ticker
        import matplotlib

        cur_pal = sns.color_palette()

        plt.rcParams.update(
            {'figure.figsize': (10, 8), 'figure.subplot.hspace': 0, 'figure.subplot.wspace': .2, 'axes.titlesize': 16,
            'axes.labelsize': 18,
            'xtick.labelsize': 16, 'ytick.labelsize': 16, 'lines.linewidth': 4,'figure.figsize': (15, 8), 'figure.subplot.hspace': .2, 'figure.subplot.wspace': 0, 'ytick.major.pad': 10})

        fig = plt.figure()

        fig.add_subplot(2, 3, 1)

        tau = int(input('Select best time lag tau for rf mapping [in ms]: '))
        frame = int(10 - tau / 10)
        x1 = int(input('And the pixel borders: x1: '))
        x2 = int(input('And the pixel borders: x2: '))
        y1 = int(input('And the pixel borders: y1: '))
        y2 = int(input('And the pixel borders: y2: '))

        im = plt.imshow(sta[frame, :, :][x1:x2, y1:y2], interpolation='none',
                        cmap=plt.cm.coolwarm, extent=(y1, y2, x2, x1), origin='upper')
        cbi = plt.colorbar(im)
        plt.xticks([])
        plt.yticks([])
        tick_locator = ticker.MaxNLocator(nbins=6)
        cbi.locator = tick_locator
        cbi.update_ticks()

        fig.add_subplot(2, 2, 2)
        deltat = 1000  # in ms
        t = np.linspace(100, -deltat, len(kernel))
        if np.sign(np.mean(kernel)) == -1:
            plt.plot(t, kernel, color=cur_pal[0])
        else:
            plt.plot(t, kernel, color=cur_pal[2])

        plt.locator_params(axis='y', nbins=4)
        ax = fig.gca()
        ax.set_xticklabels([])
        ax.set_xlim([100, -deltat])
        plt.ylabel('stimulus intensity', labelpad=20)

        fig.add_subplot(2, 3, 4)
        im = plt.imshow(v.reshape(sta.shape[1], sta.shape[2])[x1:x2, y1:y2], interpolation='none',
                        cmap=plt.cm.coolwarm, extent=(y1, y2, x2, x1), origin='upper')
        cbi = plt.colorbar(im)
        plt.xticks([])
        plt.yticks([])
        tick_locator = ticker.MaxNLocator(nbins=6)
        cbi.locator = tick_locator
        cbi.update_ticks()
        plt.xticks([])
        plt.yticks([])

        fig.add_subplot(2, 2, 4)

        if np.sign(np.mean(u)) == -1:
            plt.plot(t, u, color='b')
        else:
            plt.plot(t, u, color='r')

        plt.locator_params(axis='y', nbins=4)
        ax = fig.gca()
        ax.set_xlim([100, -deltat])
        plt.xlabel('time [ms]', labelpad=10)
        plt.ylabel('stimulus intensity', labelpad=20)

        return fig

    def chirp(psth, f_norm, loop_duration, delT=.1):

        # Plotting parameter
        plt.rcParams.update({'xtick.labelsize': 16, 'ytick.labelsize': 16, 'axes.labelsize': 16, 'axes.titlesize': 20,
                             'figure.figsize': (10, 8), 'lines.linewidth': 2})

        # define stimulus

        ChirpDuration = 8  # Time (s) of rising/falling chirp phase
        ChirpMaxFreq = 8  # Peak frequency of chirp (Hz)
        IntensityFrequency = 2  # freq at which intensity is modulated

        SteadyOFF = 3.00  # Time (s) of Light OFF at beginning at end of stimulus
        SteadyOFF2 = 2.00
        SteadyON = 3.00  # Time (s) of Light 100% ON before and after chirp
        SteadyMID = 2.00  # Time (s) of Light at 50% for steps

        Fduration = 0.017  # Single Frame duration (s) -  ADJUST DEPENDING ON MONITOR
        Fduration_ms = 17.0  # Single Frame duration (ms) - ADJUST DEPENDING ON MONITOR

        KK = ChirpMaxFreq / ChirpDuration  # acceleration in Hz / s
        KK2 = IntensityFrequency

        StimDuration = SteadyOFF2 + SteadyON + 2 * SteadyOFF + 3 * SteadyMID + 2 * ChirpDuration

        def stimulus():
            t = np.linspace(0, ChirpDuration, ChirpDuration / Fduration)
            Intensity0 = np.sin(3.141 * KK * np.power(t, 2)) * 127 + 127
            RampIntensity = 127 * t / (ChirpDuration)
            Intensity1 = np.sin(2 * 3.141 * KK2 * t) * RampIntensity + 127

            n_off = SteadyOFF / Fduration
            n_off2 = SteadyOFF2 / Fduration
            n_on = SteadyON / Fduration
            n_midi = SteadyMID / Fduration
            n_chirp = ChirpDuration / Fduration

            t_on = n_off2
            t_off0 = n_off2 + n_on
            t_midi0 = n_off2 + n_on + n_off
            t_chirp0 = n_off2 + n_on + n_off + n_midi
            t_midi1 = n_off2 + n_on + n_off + n_midi + n_chirp
            t_chirp1 = n_off2 + n_on + n_off + n_midi + n_chirp + n_midi
            t_midi2 = n_off2 + n_on + n_off + n_midi + n_chirp + n_midi + n_chirp
            t_off1 = n_off2 + n_on + n_off + n_midi + n_chirp + n_midi + n_chirp + n_midi

            tChirp = np.linspace(0, StimDuration, StimDuration / Fduration)
            chirp = np.zeros(len(tChirp))

            chirp[t_on:t_off0 - 1] = 255
            chirp[t_midi0:t_chirp0] = 127
            chirp[t_chirp0:t_midi1] = Intensity0
            chirp[t_midi1:t_chirp1] = 127
            chirp[t_chirp1:t_midi2 - 1] = Intensity1
            chirp[t_midi2:t_off1] = 127

            return tChirp, chirp

        T = int(loop_duration)  # in s

        nbins1 = T / delT

        tPSTH = np.linspace(0, T, nbins1)

        ntrials = len(f_norm)
        fig, axarr = plt.subplots(3, 1, sharex=True)
        plt.subplots_adjust(hspace=.7)

        for trial in range(ntrials):
            axarr[1].scatter(f_norm[trial] / 10000, trial * np.ones([len(f_norm[trial] / 10000)]),
                             color='k')  # scatter(tStar{trial},trial*ones(1,length(tStar{trial})),'b.')
            axarr[1].set_ylabel('# trial', labelpad=20)

        axarr[1].set_ylim(-.5, ntrials - .5)
        axarr[1].set_yticklabels(np.linspace(0, ntrials, ntrials + 1).astype(int))

        axarr[2].plot(tPSTH, psth, 'k')
        axarr[2].set_ylabel('PSTH', labelpad=10)
        axarr[2].set_yticks([0, max(psth) / 2, max(psth)])
        axarr[2].set_xlabel('time [s]')

        (tChirp, chirp) = stimulus()

        axarr[0].plot(tChirp, chirp, 'k')
        axarr[0].set_ylabel('stimulus intensity', labelpad=5)
        axarr[0].set_yticks([0, 127, 250])
        axarr[0].set_xlim(0, loop_duration)

        return fig

    def ds(hist, deg):
        plt.rcParams.update({'xtick.labelsize': 16, 'ytick.labelsize': 16, 'axes.labelsize': 16, 'axes.titlesize': 20,
                             'figure.figsize': (10, 8)})

        with sns.axes_style('whitegrid'):
            fig = plt.figure()
            ax = plt.axes(polar=True, axisbg='white')
            width = .2
            rads = np.radians(deg) - width / 2
            counts = np.mean(hist, 0)
            plt.bar(rads, counts, width=width, facecolor='k')

            ycounts = [round(max(counts) / 2), round(max(counts))]
            ax.set_theta_direction(-1)
            ax.set_theta_offset(np.pi / 2)
            ax.set_yticks(ycounts)
            ax.grid(color='k', linestyle='--')
            # ax.legend('mean number of spikes',fontsize=14,loc='lower left')

        return fig, ax

    def ds_traces(voltage_trace, triggertimes, rec_type, fs=10000):


        plt.rcParams.update({'xtick.labelsize': 16, 'ytick.labelsize': 16, 'axes.labelsize': 16, 'axes.titlesize': 20,
                             'figure.figsize': (10, 8),'lines.linewidth':2,'figure.subplot.hspace': .2,'figure.subplot.hspace': .2})

        # stimulus parameter

        t_off = 2 * fs  # in s * fs
        t_on = 2 * fs  # in s

        nconditions = 8
        ntrials = int(np.floor(len(triggertimes) / nconditions))

        idx = np.array([0, 4, 6, 2, 5, 1, 7, 3])
        deg = np.arange(0, 360, 360 / 8).astype(int)

        true_loop_duration = []
        for trial in range(1, ntrials):
            true_loop_duration.append(triggertimes[trial * nconditions] - triggertimes[(trial - 1) * nconditions])
        # loop_duration_n = np.ceil(np.mean(true_loop_duration))

        stim = np.zeros(voltage_trace.shape)

        for i in range(len(triggertimes)):
            stim[triggertimes[i]:triggertimes[i] + t_on] = 1

        v_trace_trial = []
        stim_trial = []
        for i in range(len(triggertimes)):
            v_trace_trial.append(np.array(voltage_trace[triggertimes[i]:triggertimes[i] + t_on + t_off]))
            stim_trial.append(np.array(stim[triggertimes[i]:triggertimes[i] + t_on + t_off]))

        plt.rcParams.update({'figure.subplot.hspace': .1, 'figure.figsize': (15, 8)})
        N = len(v_trace_trial)

        if rec_type == 'extracell':

            scale = np.max(voltage_trace) - np.min(voltage_trace)
            offset = np.min(voltage_trace)

            fig1, axarr = plt.subplots(int(N / nconditions) + 1, nconditions, sharex=True,
                                       sharey=True)  # len(triggertimes)
            for i in range(N + nconditions):
                rowidx = int(np.floor(i / nconditions))
                colidx = int(i - rowidx * nconditions)
                if rowidx == 0:
                    axarr[rowidx, colidx].plot(stim_trial[i] * scale * .6 + offset * .9, 'k')
                    axarr[rowidx, colidx].set_xticks([])
                    axarr[rowidx, colidx].set_yticks([])
                else:
                    axarr[rowidx, colidx].plot(v_trace_trial[i - nconditions], 'k')
                    axarr[rowidx, colidx].set_xticks([])
                    axarr[rowidx, colidx].set_yticks([])
            plt.suptitle('Traces sorted by trial (row) and direction (columns)', fontsize=20)

            fig2, ax = plt.subplots()

        # Heatmap
        if rec_type == 'intracell':

            # first figure

            fig1, axarr = plt.subplots(int(N / nconditions) + 1, nconditions, sharex=True, sharey=True)
            scale = np.max(voltage_trace) - np.min(voltage_trace)
            offset = np.min(voltage_trace)

            for i in range(N + nconditions):
                rowidx = int(np.floor(i / nconditions))
                colidx = int(i - rowidx * nconditions)
                if rowidx == 0:
                    axarr[rowidx, colidx].plot(stim_trial[i] * scale * .6 + offset * .9, 'k')
                    axarr[rowidx, colidx].set_xticks([])
                    axarr[rowidx, colidx].set_yticks([])
                else:
                    axarr[rowidx, colidx].plot(v_trace_trial[i - nconditions], 'k')
                    axarr[rowidx, colidx].set_xticks([])
                    axarr[rowidx, colidx].set_yticks([])
            plt.suptitle('Traces sorted by trial (row) and direction (columns)', fontsize=20)

            # second figure

            fig2, ax = plt.subplots()

            arr = np.array(v_trace_trial)
            arr = arr.reshape(ntrials, nconditions, arr.shape[1])

            for trial in range(ntrials):
                arr[trial, :, :] = arr[trial, :, :][idx]

            l = []
            for cond in range(nconditions):
                for trial in range(ntrials):
                    l.append(arr[trial, cond, :])

            intensity = np.array(l).reshape(ntrials, nconditions, len(l[0]))

            column_labels = np.linspace(0, 4, 5)
            row_labels = deg.astype(int)

            plt.pcolormesh(np.mean(intensity, 0), cmap=plt.cm.coolwarm)
            cax = plt.colorbar()
            cax.set_label('voltage [mV]', rotation=270, labelpad=50)
            plt.xlabel('time [s]')
            plt.ylabel('direction [deg]')
            plt.title('Average membrane potential')

            ax.set_xticks(np.linspace(0, len(l[0]), 5), minor=False)
            ax.set_yticks(np.arange(intensity.shape[1]) + .5, minor=False)

            ax.invert_yaxis()
            ax.xaxis.set_ticks_position('bottom')

            ax.set_xticklabels(column_labels, minor=False)
            ax.set_yticklabels(row_labels, minor=False)
        else:
            print('Unknown recording type')
        return fig1, fig2

    def on_off(voltage_trace, triggertimes, rec_type, fs=10000):

        plt.rcParams.update({'xtick.labelsize': 16, 'ytick.labelsize': 16, 'axes.labelsize': 16, 'axes.titlesize': 20,
                             'figure.figsize': (10, 8), 'lines.linewidth': 2, 'figure.subplot.hspace': .2,
                             'figure.subplot.hspace': .2})

        t_off = .5 * fs  # in s * fs
        t_on = .5 * fs  # in s

        stim = np.zeros(voltage_trace.shape)

        for i in range(len(triggertimes)):
            stim[triggertimes[i] + t_off:triggertimes[i] + t_off + 2 * t_on] = 1

        v_trace_trial = []
        stim_trial = []
        for i in range(len(triggertimes)-1):
            v_trace_trial.append(np.array(voltage_trace[triggertimes[i]:triggertimes[i] + 2 * t_off + 2 * t_on]))
            stim_trial.append(np.array(stim[triggertimes[i]:triggertimes[i] + 2 * t_off + 2 * t_on]))

        scale = np.max(voltage_trace) - np.min(voltage_trace)
        offset = np.min(voltage_trace)

        plt.rcParams.update(
            {'axes.titlesize': 20, 'axes.labelsize': 18, 'xtick.labelsize': 16, 'ytick.labelsize': 16,
             'figure.figsize': (15, 8), 'figure.subplot.hspace': .1})
        fig1, axarr = plt.subplots(4, int(np.ceil(len(triggertimes) / 2)), sharex=True, sharey=True)

        for i in range(len(v_trace_trial)):
            rowidx = 2 * int(np.ceil((i + 1) / (len(v_trace_trial) * .5)) - 1)
            colidx = int(i - (rowidx) * len(v_trace_trial) * .5)
            axarr[rowidx, colidx].plot(stim_trial[i] * scale + offset, 'k', linewidth=2)
            axarr[rowidx, colidx].set_xticks([])
            axarr[rowidx, colidx].set_yticks([])
            axarr[rowidx + 1, colidx].plot(v_trace_trial[i], 'k')
            axarr[rowidx + 1, colidx].set_xticks([])
            axarr[rowidx + 1, colidx].set_yticks([])

        # Plot heatmap

        if rec_type == 'intracell':

            fig2, ax = plt.subplots()

            intensity = np.array(v_trace_trial)

            plt.pcolormesh(intensity, cmap=plt.cm.coolwarm)

            cax = plt.colorbar()
            cax.set_label('voltage [mV]', rotation=270, labelpad=50)

            ax.set_ylim([0, len(v_trace_trial)])
            ax.set_xticks(np.linspace(0, intensity.shape[1], 5), minor=False)
            # ax.set_yticks(np.linspace(0,intensity.shape[0],5), minor=False)

            ax.invert_yaxis()
            ax.xaxis.set_ticks_position('bottom')

            # row_labels = np.linspace(0,intensity.shape[0],5)
            column_labels = np.linspace(0, 2, 5)
            ax.set_xticklabels(column_labels, minor=False)
            # ax.set_yticklabels(row_labels, minor=False)

            plt.xlabel('time [s]')
            plt.ylabel('trial')
            plt.title('Membrane potential')
        else:
            fi2, ax = plt.subplots()

        return fig1, fig2

    def overlay_center(morph, rf_center):

        plt.rcParams.update({
            'figure.figsize': (15, 8),
            'figure.subplot.hspace': .2,
            'figure.subplot.wspace': .2,
            'axes.titlesize': 20,
            'axes.labelsize': 18
        })

        line_stack = np.ma.masked_where(morph == 0, morph)

        clim = (np.min(morph), np.max(morph) * .2)
        fig, ax = plt.subplots()

        ax.imshow(rf_center, cmap=plt.cm.coolwarm)
        ax.imshow(line_stack, cmap=plt.cm.gray)
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        return fig

    def overlay_rf(morph_pad,rf_up):

        plt.rcParams.update({
                'figure.figsize':(15,8),
                'figure.subplot.hspace':.2,
                'figure.subplot.wspace':.2,
                'axes.titlesize': 20,
                'axes.labelsize': 18
            })

        line_pad = np.ma.masked_where( morph_pad == 0, morph_pad)

        clim = (np.min(morph_pad), np.max(morph_pad)*.2)

        fig,ax = plt.subplots()
        ax.imshow(rf_up,cmap = plt.cm.coolwarm)
        ax.imshow(line_pad,cmap = plt.cm.gray, clim = clim)

        ax.set_xticklabels([])
        ax.set_yticklabels([])

        return fig

    def overlay_gauss(self, morph_pad, rf_up, params_m, params_rf):

        plt.rcParams.update({
            'figure.figsize': (15, 8),
            'figure.subplot.hspace': .2,
            'figure.subplot.wspace': .2,
            'axes.titlesize': 20,
            'axes.labelsize': 18,
            'lines.linewidth': 1
        })

        fig, ax = plt.subplots()
        clim = (np.min(morph_pad), np.max(morph_pad) * .2)

        line_pad_shift = np.ma.masked_where(morph_pad == 0, morph_pad)

        fit_m_pad = self.helper.gaussian(*params_m)
        fit_rf_pad = self.helper.gaussian(*params_rf)

        ax.imshow(rf_up, cmap=plt.cm.coolwarm)
        ax.imshow(line_pad_shift, cmap=plt.cm.gray, clim=clim)
        ax.contour(fit_m_pad(*np.indices(morph_pad.shape)), cmap=plt.cm.Greens, linewidth=1)
        ax.contour(fit_rf_pad(*np.indices(rf_up.shape)), cmap=plt.cm.Purples, linewidth = 1)

        ax.set_yticklabels([])
        ax.set_xticklabels([])

        return fig

class helper:

    def gaussian(height, mu_x, mu_y, sd_x, sd_y):
        """Returns a gaussian function with the given parameters"""
        sd_x = float(sd_x)
        sd_y = float(sd_y)
        return lambda x, y: height * np.exp(-((x - mu_x) ** 2 / (sd_x ** 2) + (y - mu_y) ** 2 / (sd_y ** 2)) / 2)

    def moments(data):
        """Returns (height,mu_x, mu_y, sd_x, sd_y)
        the gaussian parameters of a 2D distribution by calculating its
        moments """
        total = data.sum()
        X, Y = np.indices(data.shape)
        mu_x = (X * data).sum() / total
        mu_y = (Y * data).sum() / total
        col = data[:, int(mu_y)]
        sd_x = np.sqrt(np.abs((np.arange(col.size) - mu_y) ** 2 * col / col.sum()).sum())
        row = data[int(mu_x), :]
        sd_y = np.sqrt(np.abs((np.arange(row.size) - mu_x) ** 2 * row / row.sum()).sum())
        height = data.max()
        return height, mu_x, mu_y, sd_x, sd_y

    def fitgaussian(self,data):
        """Returns (mu_x, mu_y, sd_x, sd_y)
        the gaussian parameters of a 2D distribution found by a fit"""
        params = self.helper.moments(data)
        errorfunction = lambda p: np.ravel(self.helper.gaussian(*p)(*np.indices(data.shape)) -
                                           data)
        p, success = scoptimize.leastsq(errorfunction, params)
        return p



