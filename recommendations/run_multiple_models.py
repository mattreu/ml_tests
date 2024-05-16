from multiprocessing import Pool
from functools import partial
from alt_rbm import RBM
from alt_matrix_factor import MatrixFactorization

def run_rbm(rbm: RBM):
    error = rbm.train(return_error=True)
    return rbm, error

def run_rbms(rbms: list):
    with Pool() as p:
        results = p.map(run_rbm, rbms)
    return results

def run_mf(mf: MatrixFactorization):
    error = mf.train()
    return mf, error

def run_mfs(mfs: list):
    with Pool() as p:
        results = p.map(run_mf, mfs)
    return results