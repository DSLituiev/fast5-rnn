#sample extract file
import numpy as np
# import pickle
import h5py
import numpy as np
import os
import sys
from sklearn.preprocessing import OneHotEncoder
import logging


SEQLEN = 6
NBASES = 4

def get_hash_dict(NBASES=4):
    BASEHASH = {"A":NBASES-4, "T":NBASES-3, "C":NBASES-2, "G":NBASES-1 }
    return BASEHASH

BASEHASH = get_hash_dict(NBASES)

def get_rev_hash_dict(hash_dict=BASEHASH):
    REV_BASEHASH = dict(zip(hash_dict.values(), hash_dict.keys()))
    return REV_BASEHASH


def dnahash(seq, seq_len=SEQLEN, NBASES=NBASES):
    BASEHASH = get_hash_dict(NBASES=NBASES)
    out = 0
    logging.debug("seq[:5]\t%s"% repr(seq[:5]) )
    if seq_len is None:
        seq_len = len(seq)

    for nn, xx in enumerate(seq.upper()):
        out += (NBASES**(seq_len -1 - nn)) * BASEHASH[xx]
    return out


def convert_to_nry(x, n=4, num_digits=None):
    num_digits0 = int(np.ceil(np.log2(x+1)/np.log2(n)))
    if num_digits is not None:
        if num_digits0 > num_digits:
            raise ValueError("to few digits, overflow!")
    else:
        num_digits = num_digits0
    num_digits = num_digits - 1
    cells = np.zeros(num_digits +1, dtype=int)
    rem = x
    for j in range(num_digits,-1,-1):
        base = n**j
        cells[ num_digits - j] = rem // base
        rem = rem % base
    return cells


def dnaunhash(sh, seqlen=6, NBASES=4):
    y = convert_to_nry(sh, n=NBASES, num_digits=seqlen)
    REV_BASEHASH = get_rev_hash_dict( get_hash_dict(NBASES) )
    return "".join([REV_BASEHASH[x] for x in y])

def valid_transitions(y):
    return [ y[1:] + x for x in list("ATCG")]

def valid_transitions_hash(y, seq_len=6, NBASES=4):
    return [ NBASES*(y % NBASES**(SEQLEN-1)) + x for x in range(NBASES)]

def valid_rev_transitions(y):
    return [ x + y[:-1] for x in list("ATCG")]

def valid_rev_transitions_hash(y, seq_len=6, NBASES=4):
    return [ (y // NBASES) + x * NBASES**(seq_len-1)  for x in range(NBASES)]

def get_transition_matrix(NBASES=NBASES, SEQLEN=SEQLEN):
    hashsize = NBASES**SEQLEN
    transition_matrix = np.zeros((hashsize, hashsize), dtype=int)
    for nn in range(NBASES**SEQLEN):
        transition_matrix[nn, valid_transitions_hash(nn)] = 1
    return transition_matrix

def get_rev_transition_matrix(NBASES=NBASES, SEQLEN=SEQLEN):
    hashsize = NBASES**SEQLEN
    transition_matrix = np.zeros((hashsize, hashsize), dtype=int)
    for nn in range(NBASES**SEQLEN):
        transition_matrix[nn, valid_rev_transitions_hash(nn)] = 1
    return transition_matrix

def crop_hash(y, fro=2, to = 4):
    return (y % 4**to)  // 4**fro

def hash_window_mapping(fro=2, to=4, n_input_states = None, seq_len=SEQLEN, nbases=NBASES):
    """creates a windowing mapping matrix, e.g.:
    AT[CG]GA -> CG
    TC[GA]CG -> GA
    ...
    """
    to_ = int(max(to, fro))
    fro_ = int(min(to, fro))
    to, fro = to_, fro_
    if n_input_states is not None:
        seq_len = int(np.log2(float(n_input_states))/np.log2(float(nbases)))
    if seq_len<to:
        raise ValueError( "end of the cropping window (%u) should be less than sequence length (%u)" %\
                        (to, seq_len))

    cropped_len = to-fro
    n_cropped_states = nbases**cropped_len
    if n_input_states is None:
        n_input_states = nbases**seq_len
    print("M: %u nt --> %u nt" % ( int(np.log2(float(n_input_states))/np.log2(float(nbases))), cropped_len), file=sys.stderr)
    output_dim = (n_input_states, n_cropped_states )
    M = np.zeros(output_dim)
    for n in range(output_dim[0]):
        M[n, crop_hash(n, fro=fro, to=to)] = 1
    return M


if __name__ == "__main__":
    assert 0==np.sum((get_rev_transition_matrix() - get_transition_matrix().T)**2), "tranisition matrix is asymmetric!"
