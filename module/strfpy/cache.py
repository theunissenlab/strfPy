
# Converted using ChatGPT from df_checksum.m, df_create_stim_cache_file.m, df_checksum_with_md5.m
# 20230404

import os
import hashlib
import tempfile
import numpy as np

def df_dir_of_caches():
    return "null", 10

def df_create_stim_cache_file(outputPath, DS):
    """Creates the file with the hashes of the stimuli"""
    hashes_of_stims = []
    if not os.path.exists(os.path.join(outputPath, 'stim_hashes')):
        os.makedirs(os.path.join(outputPath, 'stim_hashes'))
    running_flag = True
    using_bar = False
    for ii in range(len(DS)):
        dsname = DS[ii]['stimfiles']
        dsdir, dsname = os.path.split(dsname)
        filename = os.path.join(outputPath, 'stim_hashes', dsname)
        if not os.path.exists(filename):
            if running_flag and not using_bar:
                using_bar = True
            this_hash = hashlib.md5(open(DS[ii]['stimfiles'], 'rb').read()).hexdigest()
            with open(filename, 'w') as f:
                f.write(this_hash)
        else:
            with open(filename, 'r') as f:
                this_hash = f.read().strip()
        hashes_of_stims.append(this_hash)
    return hashes_of_stims


def df_create_spike_cache_file(outputPath, DS):
    hashes_of_spikes = []
    if not os.path.exists(os.path.join(outputPath, 'spike_hashes')):
        os.makedirs(os.path.join(outputPath, 'spike_hashes'))
    for ii in range(len(DS)):
        dsname = DS[ii]['respfiles']
        _, dsname_ext = os.path.splitext(dsname)
        dsname = dsname + dsname_ext
        filename = os.path.join(outputPath, 'spike_hashes', dsname)
        if not os.path.isfile(filename):
            with open(DS[ii]['respfiles'], 'rb') as f:
                this_hash = hashlib.sha256(f.read()).hexdigest()
            with open(filename, 'w') as f:
                f.write(this_hash)
        else:
            with open(filename, 'r') as f:
                this_hash = f.read().strip()
        hashes_of_spikes.append(this_hash)
    return hashes_of_spikes


def df_checksum(*args):
    """
    Produces a checksum of the input arguments, which can be anything.
    Note: this is NOT a hash; it does not have crypto strength.  BUT it's
    still pretty sensitive on its inputs, so it will do for our purposes.
    """
    if os.path.exists('/usr/bin/openssl'):
        return df_checksum_with_md5(*args)
    else:
        input = concat_for_checksum(args)
        filter = np.array([8, 28, -84, 1, 69, 114, -45, -49, 107, -55, 92, 118, -55, -53, -102, -94])
        the_conv = np.convolve(filter, np.fromiter(input, dtype=int))
        out = ''
        for jj in range(16):
            toadd = np.mod(np.sum(the_conv[jj::16]), 256)
            out += format(toadd, '02x')
        return out

def df_checksum_with_md5(*args):
    testnum = 0
    done = False
    while not done:
        tempfile = os.path.join(os.getenv('TMP'), f'Temp_hashing_name_{testnum}.tmp')
        if not os.path.exists(tempfile):
            with open(tempfile, 'wb') as f:
                for arg in args:
                    f.write(str(arg).encode())
            md5 = hashlib.md5()
            with open(tempfile, 'rb') as f:
                while True:
                    data = f.read(4096)
                    if not data:
                        break
                    md5.update(data)
            out = md5.hexdigest().upper()
            done = True
        else:
            testnum += 1
    os.remove(tempfile)
    return out

def concat_for_checksum(args):
    """
    Converts all input arguments to a byte string and concatenates them.
    """
    bstr = b''
    for arg in args:
        if isinstance(arg, str):
            bstr += arg.encode()
        elif isinstance(arg, bytes):
            bstr += arg
        else:
            bstr += str(arg).encode()
    return bstr
