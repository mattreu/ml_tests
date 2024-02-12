from multiprocessing import Pool
from alt_rbm import RBM

def run_rbm(rbm: RBM):
    error = rbm.train(return_error=True)
    return rbm, error

def run_rbms(rbms: list):
    with Pool() as p:
        results = p.map(run_rbm, rbms)
    return results