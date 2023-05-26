
from strfpy.trnDirectFit import trnDirectFit

if __name__ == "__main__":
    import pickle
    tmp_modelParams = pickle.load(open("/tmp/modelParams.pkl",'rb'))
    tmp_datIdx = pickle.load(open("/tmp/datIdx.pkl",'rb'))
    tmp_optOptions = pickle.load(open("/tmp/optOptions.pkl",'rb'))
    tmp_globDat = pickle.load(open("/tmp/globDat.pkl",'rb'))
    trnDirectFit(tmp_modelParams, tmp_datIdx, tmp_optOptions, tmp_globDat, 1)