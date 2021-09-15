# %%
import sys, os
import numpy as np
#import matplotlib.pyplot as plt

import yaml
import argparse
import pprint
from scipy import ndimage
from tqdm import tqdm

# %%
class BarkleySimluation3D:
    def __init__(self, a=0.6, b=0.01, epsilon=0.02, deltaT=0.01, deltaX=0.1, D=0.02, alpha=1, boundary_mode='noflux'):
        self.a = a
        self.b = b
        self.epsilon = epsilon
        self.boundary_mode = boundary_mode # Neumann
        self.deltaT = deltaT
        self.deltaX = deltaX
        self.alpha = alpha
        self.h = D/self.deltaX**2
                    
    def set_boundaries(self, oldFields):
        if self.boundary_mode == "noflux":
            for (field, oldField) in zip((self.u, self.v), oldFields):
                field[:,:,0] = oldField[:,:,1]
                field[:,:,-1] = oldField[:,:,-2]
                
                field[:,0,:] = oldField[:,1,:]
                field[:,-1,:] = oldField[:,-2,:]
                
                field[0,:,:] = oldField[1,:,:]
                field[-1,:,:] = oldField[-2,:,:]
        
    def explicit_step(self):
        uOld = self.u.copy()
        vOld = self.v.copy()

        f = 1/self.epsilon * self.u * (1 - self.u) * (self.u - (self.v+self.b)/self.a)
        
        laplace = -6*self.u.copy()

        laplace += np.roll(self.u, +1, axis=0)
        laplace += np.roll(self.u, -1, axis=0)
        laplace += np.roll(self.u, +1, axis=1)
        laplace += np.roll(self.u, -1, axis=1)
        laplace += np.roll(self.u, +1, axis=2)
        laplace += np.roll(self.u, -1, axis=2)

        self.u = self.u + self.deltaT * (f + self.h * laplace)
        self.v = self.v + self.deltaT * (np.power(uOld, self.alpha) - self.v)

        self.set_boundaries((uOld, vOld))


def get_starting_condition_chaotic(size=[120,120,120], seed=42):
    def initialize_random(n_boxes=(20,20,20), size=(120,120,120), seed=None):
        np.random.seed(seed)
        tmp = np.random.rand(*n_boxes)
        
        rpt = size[0]//n_boxes[0], size[1]//n_boxes[1], size[2]//n_boxes[2]
        
        tmp = np.repeat(tmp, np.ones(n_boxes[0], dtype=int)*rpt[0], axis=0)
        tmp = np.repeat(tmp, np.ones(n_boxes[1], dtype=int)*rpt[1], axis=1)
        tmp = np.repeat(tmp, np.ones(n_boxes[2], dtype=int)*rpt[2], axis=2)
        
        U = tmp
        
        V = U.copy()
        V[V<0.4] = 0.0
        V[V>0.4] = 1.0
        
        return U, V
    
    U, V = initialize_random(size=size, seed=seed)
    return U, V
    
def simulate_barkley(a=0.75, b=0.06, epsilon=0.08, alpha=3, 
                     starting_condition="chaotic", dt=0.01, ds=0.1, D=0.02, size=[120,120,120], 
                     dSave=4, max_save_length=512, num_sims=8, dataset="regimeB", seed=42, init_phase=0):
 
    seeds = np.random.randint(0,1000000, 2)

    u0, v0 = get_starting_condition_chaotic(size=size, seed=seeds[0])
        
    s=BarkleySimluation3D(a=a, b=b, epsilon=epsilon, deltaT=dt, deltaX=ds, D=D, alpha=alpha)
    s.u = u0
    s.v = v0

    transform = lambda data: (np.array(data)*255-128).astype(np.int8)

    if init_phase=='random':
        init_phase = np.random.randint(3000,3500)
    elif isinstance(init_phase, int):
        pass


    print('Init phase')
    for i in tqdm(range(init_phase)):
        s.explicit_step()

    U = []
    num_sim = 0

    print('Start recording')
    for i in tqdm(range(max_save_length*dSave*num_sims+num_sims*dSave)):
        #import IPython ; IPython.embed() ; exit(1)
        s.explicit_step() 

        if i%dSave==0:
            U.append(s.u)
            if len(U)>=max_save_length:                    
                U = transform(U)
                savename = f'{seeds[0]}_{seeds[1]}_{num_sim:02d}'
                np.save(savename, U)

                del U
                U = []
                num_sim += 1
        
if __name__=='__main__':
    simulate_barkley()