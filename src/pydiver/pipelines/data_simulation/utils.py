# %%
import sys, os
import numpy as np
#import matplotlib.pyplot as plt

import yaml
import argparse
import pprint
from scipy import ndimage

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

def get_starting_condition_two_spirals(size=[120,120,120], seeds=[0,1]):
    def getRotatedSpiral(size=[60,60,60], seed=None):
        np.random.seed(seed=seed)

        rot_size = np.array(size)*5
        u, v = np.zeros((2, *rot_size))

        spiralPosX = np.random.randint(size[0]*0.25, size[0]*0.75)
        spiralPosY = np.random.randint(size[1]*0.25, size[1]*0.75)

        x, y = spiralPosX+int(2*size[0]), spiralPosY+int(2*size[1])

        u[x:,:,:] = 1.
        v[:,:y,:] = 0.5

        p1,p2,p3 = np.random.randint(-90,90,3)
        u = ndimage.rotate(u, p1, axes=(0,1), reshape=0)
        u = ndimage.rotate(u, p2, axes=(0,2), reshape=0)
        u = ndimage.rotate(u, p3, axes=(1,2), reshape=0)

        v = ndimage.rotate(v, p1, axes=(0,1), reshape=0)
        v = ndimage.rotate(v, p2, axes=(0,2), reshape=0)
        v = ndimage.rotate(v, p3, axes=(1,2), reshape=0)

        s = u.shape
        u = u[s[0]//2-size[0]//2:s[0]//2+size[0]//2,s[1]//2-size[1]//2:s[1]//2+size[1]//2,s[2]//2-size[2]//2:s[2]//2+size[2]//2]
        v = v[s[0]//2-size[0]//2:s[0]//2+size[0]//2,s[1]//2-size[1]//2:s[1]//2+size[1]//2,s[2]//2-size[2]//2:s[2]//2+size[2]//2]

        u[u>0.5]=1
        u[u<=0.5]=0
        v[v>0.4]=0.5
        v[v<=0.4]=0.0

        return u, v
    
    size = (np.array(size)/2).astype(int)
    
    _u, _v = getRotatedSpiral(size=size, seed=seeds[0])
    _u2, _v2 = getRotatedSpiral(size=size, seed=seeds[1])
    
    U = np.concatenate([_u, np.zeros(size)], axis=2)
    U = np.concatenate([U, np.zeros(size*[1,1,2])], axis=1)
    U2 = np.concatenate([np.zeros(size), _u2], axis=2)
    U2 = np.concatenate([np.zeros(size*[1,1,2]), U2], axis=1)
    U = np.concatenate([U, U2], axis=0)
    
    V = np.concatenate([_v, np.zeros(size)], axis=2)
    V = np.concatenate([V, np.zeros(size*[1,1,2])], axis=1)
    V2 = np.concatenate([np.zeros(size), _v2], axis=2)
    V2 = np.concatenate([np.zeros(size*[1,1,2]), V2], axis=1)
    V = np.concatenate([V, V2], axis=0)
    
    return U, V
    
def simulate_barkley(a=0.6, b=0.01, epsilon=0.02, alpha=1, 
                     starting_condition="two_spirals", dt=0.01, ds=0.1, D=0.02, size=[120,120,120], 
                     dSave=16, max_save_length=32, num_sims=16, dataset="regimeA", seed=42):
    #print("START!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
 
    seeds = np.random.randint(0,1000000, 2)

    if starting_condition=='two_spirals':
        u0, v0 = get_starting_condition_two_spirals(size=size, seeds=seeds)
    elif starting_condition=='chaotic':
        u0, v0 = get_starting_condition_chaotic(size=size, seed=seeds[0])
        
    savename = f'{seeds[0]}_{seeds[1]}'
        
    s=BarkleySimluation3D(a=a, b=b, epsilon=epsilon, deltaT=dt, deltaX=ds, D=D, alpha=alpha)
    s.u = u0
    s.v = v0
    
    U = []

    transform = lambda data: (np.array(data)*255-128).astype(np.int8)

    init_phase = 10 #np.random.randint(3000,3500)
    

    for i in range(100000):
        #import IPython ; IPython.embed() ; exit(1)
        s.explicit_step()
    
        if i>=init_phase:
            if i==init_phase: 
                print('Start recording')
            if i%dSave==0:
                U.append(s.u)
                if len(U)>=max_save_length:                    
                    U = transform(U)
                    return {savename:U}
    return {savename:U}

def main(kwargs={}):
    return simulate_barkley(**kwargs)
        