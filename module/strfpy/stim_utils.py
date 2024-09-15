
import os
import re
import scipy.io as sio
import numpy as np
# from config_ECoG import  WAV_DIR, OUTPUT_DATA_PATH, TRFILE_DIR, CHUNK, FEATURES

# import BOLD_MPS.utils
# from BOLD_MPS import utils
# from BOLD_MPS.utils.sound_utils import WavPrep
# from BOLD_MPS.utils.utils import save_table_file, load_hf5
# from BOLD_MPS.utils.features_utils import mps_preproc

def load_stiminfo(stimfile):
    """ Loads the stimulus info structure
    """

    matdata = sio.loadmat(stimfile)["sentdet"][0]

    # load in field names from fieldnames.txt
    with open(f"fieldnames.txt", "r") as f:
        fieldnames = f.read().splitlines()

    # create dictionary of fieldnames
    field_dict = {fieldnames[i]:i for i in range(len(fieldnames))}

    sent_struct = dict()
    for i in range(len(matdata)):
        
        # neural response in entry 
        sentid = matdata[i][field_dict['name']][0]

        # data sampling rate,  num trials
        sound =  np.array(matdata[i][field_dict['sound']])
        soundf = matdata[i][field_dict['soundf']][0]
        dataf =  matdata[i][field_dict['dataf']][0] 
        aud = np.array(matdata[i][field_dict['aud']])  
        duration = np.array(matdata[i][field_dict['duration']])
        loudness = np.array(matdata[i][field_dict['loudness']])
        wordlist = np.array(matdata[i][field_dict['wordList']])
        syltype =  np.array(matdata[i][field_dict['syltype']])
        wordOns = np.array(matdata[i][field_dict['wordOns']])

        # time before baseline, after,
        befaft = np.array(matdata[i][field_dict['befaft']][0]) 
        
        # out structure is indexed by sentences
        tmp = {'sound': sound, 'soundf': soundf, 'dataf': dataf, 'aud': aud, 
               'befaft': befaft, 'duration': duration, 'loudness': loudness, 
               'wordList': wordlist, 'syltype': syltype, 'wordOns': wordOns}
        sent_struct[sentid] = tmp
    return sent_struct, field_dict


def load_preprocstim(to_plot=False, chunk:str=CHUNK, feature:str=FEATURES[0]):
    '''Load in stimulus features and segment into stimulus locked chunks
    for training and testing sets
    
    Calculate MPS and spectrogram features for each chunk

    '''

    # run MPS and spectrogram calculation on chunked TIMIT
    wav_paths = [f for f in os.listdir(WAV_DIR) if "wav" in f]
    MPS_DIR =  f"{OUTPUT_DATA_PATH}/{chunk}"
    
    print ("load in chunk segment... " + chunk.upper())
    ctr = 0
    for p in wav_paths:
        ctr+=1
        
        fname = p.split(".")[0].split("-")[0]
        # see if file ends with number
        m = re.search(r'\d+$', fname) 
        
        output_path_file = f"{OUTPUT_DATA_PATH}/{chunk}/{fname}_{chunk}_soundProc.hf5"
        if m is not None and not os.path.isfile(output_path_file):   
        
            msg = str(ctr) + ': saving BioSound for ' + fname
            print(msg)

            # load in wav file
            w_load = WavPrep(f"{WAV_DIR}/{p}", chunk, allstories=[fname], 
                                        grid_dir=WAV_DIR, trfile_dir=TRFILE_DIR)

            # segment wav file into chunks
            w_load.wav_chunk(w_load.time_starts, w_load.time_ends)
            w_load.wav_resample()
            w_load.wav_padding()
            w_load.wav_norm()
            w_load.generate_biosound()

            # save out biosound features
            mps_all = np.array([i.mps for i in w_load.biosound])
            spectro_all = np.array([i.spectro for i in w_load.biosound])
            label_all = w_load.label
            fo = w_load.biosound[0].fo
            to = w_load.biosound[0].to
            wt = w_load.biosound[0].wt
            wf = w_load.biosound[0].wf
            time_starts = w_load.time_starts
            time_ends = w_load.time_ends
            
            save_table_file(output_path_file, dict(mps=mps_all, spectro=spectro_all, label=label_all,
                                                        fo=fo, to=to, wt=wt, wf=wf, time_starts=time_starts, 
                                                        time_ends=time_ends))

    ## NOTE: currently can only load one feature at a time
    print ("load in stimulus features... " + feature.upper())
    mps_load = load_hf5(MPS_DIR, feature)

    ## preproc data: demean (optional) & reshape
    mps_proc_tmp = mps_preproc(mps_load, run_demean=True) # , run_crop=True, 
                               # crop_start=45, crop_end=56

    ## change keys to be sentence names without file info, e.g. diphone, hf5
    mps_proc = dict()
    for key, value in mps_proc_tmp.items():
        mps_proc['_'.join(key.split('_')[0:2])] = value

    if to_plot:
        # load in wav file
        w_load = WavPrep(f"{WAV_DIR}/{wav_paths[0]}", chunk, allstories=[wav_paths[0].split(".")[0].split("-")[0]], 
                                    grid_dir=WAV_DIR, trfile_dir=TRFILE_DIR)

        # check chunk visually
        w_load.biosound[0].spectrum(f_high=5000)
        w_load.biosound[0].meantime = w_load.biosound[0].stdtime =w_load.biosound[0].kurtosistime = 0
        w_load.biosound[0].skewtime = w_load.biosound[0].entropytime = 0
        w_load.biosound[0].plot(DBNOISE=50, f_low=10, f_high=5000)
    
    ## return preproc data
    return mps_proc

