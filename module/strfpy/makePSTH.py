
import numpy as np

class SpikeResponse():
    
    def __init__(self, fileName):
        
        self.fileName = fileName

        self.spikeTrials = self._read_spikes_from_file()
        self.resp = np.asanyarray([])


    def _read_spikes_from_file(self):
        spikeTimes = np.loadtxt(self.fileName, delimiter=' ')
        spikeTrials = []
        for j in range(len(spikeTimes)):
            spikeTrials.append(spikeTimes[j, spikeTimes[j, :] > 0])
        
        return spikeTrials


    def preprocess_response(self, stimLength, sampleRate):


        nSpikeTrials = len(self.spikeTrials)
        spikeIndicies = [None]*nSpikeTrials
        for j in range(nSpikeTrials):
            stimes = self.spikeTrials[j]
            # turn spike times (ms) into indexes at response sample rate
            stimes = np.round(stimes * 1e-3 * sampleRate).astype(int)
            # remove excess zeros
            stimes = stimes[stimes > 0]
            spikeIndicies[j] = stimes

        psth = self.make_psth(self.spikeTrials, stimLength*1e3, 1)

        resp = {}
        resp['type'] = 'psth'
        resp['sampleRate'] = sampleRate
        resp['rawSpikeTimes'] = self.spikeTrials
        resp['rawSpikeIndicies'] = spikeIndicies
        resp['psth'] = psth

        self.resp = resp


    #  converted
    def make_psth(self, stimdur, binsize):

        nbins = round(stimdur / binsize)
        psth = np.zeros(nbins)

        ntrials = len(self.spikeTrials)

        maxIndx = round(stimdur / binsize)

        for k in range(ntrials):
            stimes = self.spikeTrials[k]
            indx = np.logical_and(stimes > 0, stimes < stimdur)

            stimes = stimes[indx]
            sindxs = np.round(stimes / binsize).astype(int) + 1

            sindxs[sindxs == 0] = 1
            sindxs[sindxs > maxIndx] = maxIndx

            psth[sindxs - 1] += 1

        psth /= ntrials
        return psth


