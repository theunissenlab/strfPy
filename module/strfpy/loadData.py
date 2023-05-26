

class SRData():
    """

    prepross .wav files and spike tiumes into time-frequency representations and PSTHs
    
    
    ----------
    rawStimFiles : 
        a cell array of .wav file names
    rawRespFiles : 
        a cell array of spike-time file names. 
        Each file contains a space-separated list of file names, one line for each trial.
        Each spike time is specified in milliseconds.


    Attributes
    ----------


    """

    def __init__(self, rawStimFiles):
        self.pairCount = len(rawStimFiles)
        

