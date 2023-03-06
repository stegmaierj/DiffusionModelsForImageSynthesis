# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 17:27:36 2021

@author: Nutzer
"""

import os
import glob
import torch
import numpy as np

from skimage import io, measure, morphology

from utils.h5_converter import h5_writer, h5_reader, add_group, remove_h5_group, calculate_flows
from utils.utils import print_timestamp
from ThirdParty.diffusion import GaussianDiffusionTrainer

#%% 
# Generate diffusion masks

filelist = glob.glob(r'/path/to/converted/data/segmentation_h5/*.h5')
try:
    remove_h5_group(filelist, source_group='data/diffusion_mask')
except:
    pass

# Used to simulate GOWT1 nucleoli
gen_holes=False

for num_file,file in enumerate(filelist):

    print_timestamp('Processing file {0}/{1}: {2}', (num_file+1, len(filelist), os.path.split(file)[-1]))    
        
    # Load the instance data
    data = h5_reader(file, source_group='data/boundary_decayed')
    data = data.astype(np.float32)
    
    if gen_holes:
        # add holes to CTCGOWT1 data
        data_tmp = morphology.erosion(data, morphology.ball(10))
        data_tmp = data_tmp.astype(np.uint8)
        
        data_indices = np.indices(data_tmp.shape)
        regions = measure.regionprops(data_tmp)
        for props in regions:
            num_holes = np.random.choice([0,1,2], size=1, replace=False, p=[0.2,0.5,0.3])
            coords = props.coords[np.random.randint(0,props.coords.shape[0], size=num_holes),:]
            for coord in coords:
                hole = (((data_indices[0,...]-coord[0])/np.random.randint(7,13))**2 +\
                        ((data_indices[1,...]-coord[1])/np.random.randint(7,13))**2 +\
                        ((data_indices[2,...]-coord[2])/np.random.randint(7,13))**2) <= 1
                data[hole] = -np.random.uniform(0.6,1)*data.max()/1.3
    
        data /= data.max()
    else:
        data /= data.max()
    
    # Increase final intensity range
    # Background is -1 and cells are in range [-0.7,1)
    data *= 1.7
    data[data>0] -= 0.7
    data[data==0] = -1
    
    # Save results
    add_group(file, data, target_group='data/diffusion_mask')

    