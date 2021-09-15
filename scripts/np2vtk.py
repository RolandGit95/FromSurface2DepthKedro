# %%
import os
import numpy as np
from pyevtk.hl import gridToVTK
from tqdm import tqdm
import glob

# %%

x,y,z = [np.arange(0,121,1).astype(np.int16) for _ in range(3)]

# +
file = '/home/roland/Projekte/Tests/FromSurface2DepthKedro/data/01_raw/regimeB/visualization/949641_45185.npy'
vtk_folder = '/home/roland/Projekte/Tests/FromSurface2DepthKedro/data/01_raw/regimeB/visualization/example1'

# +
data = np.load(file)

# +
if not os.path.exists(vtk_folder):
    os.makedirs(vtk_folder)   
    
for i, d in tqdm(enumerate(data)):
    _savename = os.path.join(vtk_folder, f'{i:04d}' + '.vtk')#   f'{save_directory}{i:04d}' + '.vtk'
    gridToVTK(_savename, x, y, z, cellData = {'u': d})


# %%
folder = '/home/roland/Projekte/Tests/FromSurface2DepthKedro/data/09_additional/visualization/regimeB/np'
vtk_folder = '/home/roland/Projekte/Tests/FromSurface2DepthKedro/data/09_additional/visualization/regimeB/vtk'

if not os.path.exists(vtk_folder):
    os.makedirs(vtk_folder) 
# %%
files = glob.glob(os.path.join(folder, '*.npy'))
files.sort()

x,y,z = [np.arange(0,121,1).astype(np.int16) for _ in range(3)]


# %%
def saveVTK(data, vtk_folder, start_idx=0):
    for i, d in tqdm(enumerate(data)):
        _savename = os.path.join(vtk_folder, f'{i+start_idx:04d}' + '.vtk')
        gridToVTK(_savename, x, y, z, cellData = {'u': d})


# %%
start_idx = 0

for file in files:
    data = np.load(file)
    saveVTK(data, vtk_folder, start_idx=start_idx)
    
    start_idx+=len(data)

# %%
