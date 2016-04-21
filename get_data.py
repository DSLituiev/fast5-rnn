#sample extract file
import numpy as np
# import pickle
import h5py
import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder
import logging
from dnahashing import dnahash

fX = np.float32
BASE_DIR = ""

def maybe_one_hot(x, onehot = False):
    if onehot:
        enc = OneHotEncoder(n_values=NBASES**SEQLEN)
        # yfun = lambda x: enc.fit_transform( np.array([ dnahash(y) for y in x]).reshape(-1,1)  )
        yfun = lambda x: np.array([ dnahash(y) for y in x]).reshape(-1,1)
    else:
        yfun = lambda x: x
    return yfun(x)


def preproc_event(mean, std, length):
    mean = mean / 100.0 - 0.66
    std = std - 1
    return [mean, mean*mean, std, length]

def extract_events(h5 , strand = "complement"):
    prefix_events = "/Analyses/Basecall_1D_000/BaseCalled_{0}/Events".format(strand)
    prefix_const = "/Analyses/Basecall_1D_000/Summary/basecall_1d_complement"
    index = 0.0
    logging.debug(prefix_const)
    logging.debug(prefix_events)
    try:
        tscale_sd = h5[ prefix_const ].attrs["scale_sd"]
        tscale = h5[ prefix_const ].attrs["scale"]
        tshift = h5[ prefix_const ].attrs["shift"]
        tdrift = h5[ prefix_const ].attrs["drift"]
    except KeyError as err:
        print(err, file = sys.stderr)
        return np.array(), [], 0,0,0,0
    event_tuples = []
    base_from_step = []
    events = h5[ prefix_events ]
    #print([x for x in h5[prefix_events].attrs])
    try:
        for e in events:
            mean = (e["mean"] - tshift - index * tdrift) / tscale
            stdv = e["stdv"] / tscale_sd
            length = e["length"]
            event_tuples.append(preproc_event(mean, stdv, length))
            index += e["length"]
            if e["move"] == 1:
                base_from_step.append(e["mp_state"].decode()[2])
            if e["move"] == 2:
                base_from_step.append(e["mp_state"].decode()[1:3])
    except IndexError as ee:
        print(ee)
#     print(events)
#     events = np.array(events.value, dtype=np.float32)
    return events, base_from_step, tshift, tdrift, tscale, tscale_sd

import pandas as pd

def get_strand_data(filename, strand = "complement"):
    with h5py.File(filename, "r") as h5:
        events, bases, tshift, tdrift, tscale, tscale_sd = extract_events(h5 , strand = strand)
        keys = events.dtype.fields.keys()
        events = pd.DataFrame({k: events[k] for k in keys})

    for cc in events.columns:
        if events[cc].dtype == pd.np.object:
            events[cc] = events[cc].map(lambda x: x.decode()).astype(str)
    return events, bases

def get_files(directory_name, suffix = ".fast5"):
    for dirname, _, files in os.walk(directory_name):
        for filename in files:
            if filename.endswith( suffix ):
                filename = os.path.join(dirname, filename)
                yield filename

def read_data_gen(directory_name):
    for filename in get_files(directory_name):
        events, bases = get_strand_data(filename, strand = "template")

        features = events[["length", "mean", "stdv"]].as_matrix()
        labels = events["model_state"].as_matrix()
        yield (features, labels, filename)

SEQLEN=6
NBASES = 4
BASEHASH = {"A":NBASES-4, "C":NBASES-3, "G":NBASES-2, "T":NBASES-1 }

def dnahash(seq):
    out = 0
    logging.debug("seq[:5]\t%s"% repr(seq[:5]) )
    for nn, xx in enumerate(seq.upper()):
        out += (NBASES**nn) * BASEHASH[xx]
    return out

def get_data_from_summary_file(summary_file):
    with h5py.File(summary_file, "r") as h5f:
        files = [x for x in h5f["data"].keys()]
        logging.info( "%u (x,y) sets found in %s" % (len(files), summary_file) )
        for ff in h5f["data"].keys():
            logging.debug( "set %s" % ( ff ) )
            yield  (h5f[ "/".join(["data", ff, "features"]) ].value,
                    h5f[ "/".join(["data", ff, "labels"]) ].value,
                    )


def get_batch_from_summary_file( summary_file, batch_size = 1, ):
    x_batch = []
    y_batch = []
    logging.info( "batch_size = %s" % repr(batch_size) )
    for nn, (X, Y, ) in enumerate(get_data_from_summary_file(summary_file)):
        x_batch.append(X)
        y_batch.append(Y)
        if (nn + 1) % batch_size == 0:
            out = ((x_batch), (y_batch) )
            #logging.info( "sample # %u" % (nn+1) )
            #logging.info( "\tlen = %u" % len(out) )
            yield out #(x_batch, y_batch)
            x_batch = []
            y_batch = []
    #raise StopIteration

def get_batch_chunks_from_summary_file( summary_file, chunk_length, batch_size = 1, onehot = False ):
    x_batch = []
    y_batch = []
    logging.info( "batch_size = %s" % repr(batch_size) )

    def reshape(batch):
        return np.transpose(np.dstack(batch),(2,1,0) )

    for nn, (X, Y, ) in enumerate(get_data_from_summary_file(summary_file)):
        #print( nn)
        for jj in range( len(Y) // chunk_length):
            #print( jj*chunk_length, (jj+1)*chunk_length)
            x_batch.append(X[jj*chunk_length : (jj+1)*chunk_length])
            y_batch.append(Y[jj*chunk_length : (jj+1)*chunk_length])
            if (jj + 1) % batch_size == 0:
                out = tuple(map(reshape, (x_batch, y_batch)))
                #logging.info( "sample # %u" % (nn+1) )
                #logging.info( "\tlen = %u" % len(out) )
                yield out #(x_batch, y_batch)
                x_batch = []
                y_batch = []


from scipy.sparse import csr_matrix

def get_single_chunks_from_summary_file( summary_file, chunk_length, sparse = True):
    for x,y in get_batch_chunks_from_summary_file(summary_file, chunk_length, batch_size=1, onehot=True):
        x,y = tuple(map( lambda x : np.transpose(x, (0,2,1))[0] , [x,y] ))
        #print( "y", y.shape )
        y = csr_matrix((np.ones_like(y.ravel()), (np.arange(len(y)), y.ravel() )), shape = (len(y), NBASES**SEQLEN) ).todense()
        yield x,y


def get_data( directory_name, batch_size = 1, onehot = True, gen = read_data_gen ):
    x_batch = []
    y_batch = []
    if onehot:
        enc = OneHotEncoder(n_values=NBASES**SEQLEN)
        # yfun = lambda x: enc.fit_transform( np.array([ dnahash(y) for y in x]).reshape(-1,1)  )
        yfun = lambda x: np.array([ dnahash(y) for y in x]).reshape(-1,1)
    else:
        yfun = lambda x: x

    for nn, (features, labels, _) in enumerate(gen(directory_name)):
        if batch_size==1:
            yield (features, yfun( labels) )
        else:
            x_batch.append(features)
            y_batch.append(yfun(labels))
            if (nn + 1) % batch_size == 0:
                yield x_batch, y_batch
                x_batch = []
                y_batch = []

def read_and_dump(directory_name, outfile, onehot = True):
    if onehot:
        enc = OneHotEncoder(n_values=NBASES**SEQLEN)
        # yfun = lambda x: enc.fit_transform( np.array([ dnahash(y) for y in x]).reshape(-1,1)  )
        yfun = lambda x: np.array([ dnahash(y) for y in x]).reshape(-1,1)
    else:
        yfun = lambda x: x

    with h5py.File( outfile.replace(".h5","") + '.h5', 'w') as h5f:
        for nn, (features, labels, filename) in enumerate(read_data_gen(directory_name)):
            g1 = h5f.create_group(filename)
            print(nn)
            g1.create_dataset("features", data= features.astype(np.float32))
            g1.create_dataset("labels", data= yfun(labels))
