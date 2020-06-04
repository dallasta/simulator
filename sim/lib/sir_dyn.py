import numpy as np
import pandas as pd
import random


def run_sir(seeds_idx, Tdyn, lamb, prob_rec, nprint, contacts_df, N):
    states = np.zeros([N,Tdyn], dtype=int)
    tinf = np.ones(N, dtype=int) * Tdyn
    infective = []
    N_seeds = len(seeds_idx)
    for i in range(0,N_seeds):
        infective.append(seeds_idx[i])
        tinf[seeds_idx[i]] = 0
    recovered = []
    new_inf = []

    n_inf = np.zeros(Tdyn, dtype=int)
    n_rec = np.zeros(Tdyn, dtype=int)
    n_sus = np.zeros(Tdyn, dtype=int)

    t = 0
    
    while t < Tdyn:  
        # run dynamic 
        n_inf[t] = len(infective)
        n_rec[t] = len(recovered) 
        n_sus[t] = N - len(infective) - len(recovered) 
        states[:,t] = 0
        states[infective,t] = 1
        states[recovered,t] = 2
        tmp_contact = contacts_df[contacts_df["t"] == t]
        recovered, infective = rec_nodes(prob_rec, tinf, t, infective, recovered)
        new_inf, tinf = propagate(infective, recovered, tmp_contact, lamb, N, tinf, t)
        infective = infective + new_inf
        if t % nprint == 0:
            print("t:", t, "(S,I,R)", n_sus[t], n_inf[t], n_rec[t])
        t += 1
    return n_sus, n_inf, n_rec, states, tinf


def propagate(inf, rec, contacts, lamb, N, tinf, t):
    new_inf = []
    idxi = contacts["i"].to_numpy()
    idxi = np.unique(idxi)
    idxS = list(set(idxi) - set(rec) - set(inf))
    for i in idxS:
        idxj = contacts[contacts["i"] == i]["j"]
        lamj = contacts[contacts["i"] == i]["lambda"].to_numpy()
        neigh_inf = len(np.intersect1d(idxj, inf))
        s=lamb*np.sum(lamj)
        pinf = 1 - np.exp(-s)
        if np.random.rand() < pinf:
            new_inf.append(i)
            tinf[i] = t
    return new_inf, tinf
    
def rec_nodes(prob_rec, tinf, t, inf, rec):
    for i in inf:
        if prob_rec[t-tinf[i]] > np.random.rand():
            rec.append(i)
            inf.remove(i)
            
    return rec, inf
