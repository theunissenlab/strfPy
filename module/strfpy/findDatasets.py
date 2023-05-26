import os
import numpy as np

def find_datasets(cellDir, stimDir):
    
    fnames = os.listdir(cellDir)

    stimFiles = []
    respFiles = []
    
    for fname in fnames:
        if fname.startswith('stim'):
            stimFiles.append(fname)
        if fname.startswith('spike'):
            respFiles.append(fname)
            
        
    srPairs = get_sr_files(respFiles, stimFiles, cellDir, stimDir)
                
    dataset = {'dirname': cellDir, 'srPairs': srPairs}

    return dataset

def get_sr_files(respFiles, stimFiles, cellDir, stimDir):
    
    srPairs = {}
    stimFilesGood = []
    respFilesGood = []
    
    for spikeFile in np.sort(respFiles):

        # Get the spike id for the spike file.
        idNum = spikeFile.strip('spike')

        # Find corresponding stim link file
        stimLinkFile = 'stim' + idNum

        if stimLinkFile in stimFiles:
            with open(os.path.join(cellDir,stimLinkFile), 'r') as f:
                wavFile = f.readline().strip()
            stimFilesGood.append(os.path.join(stimDir, wavFile))
            respFilesGood.append(os.path.join(cellDir, spikeFile))
        else:
            print('Warning: Cound not find stim file for spike file', spikeFile)


    srPairs['stimFiles'] = stimFilesGood
    srPairs['respFiles'] = respFilesGood
        
    return srPairs

