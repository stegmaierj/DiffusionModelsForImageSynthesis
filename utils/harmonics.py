# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 11:28:53 2020

@author: Nutzer
"""

import numpy as np
import multiprocessing as mp

from functools import partial
from skimage import morphology
from scipy.spatial import cKDTree, Delaunay
from scipy.special import sph_harm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from dipy.core.geometry import sphere2cart, cart2sphere

from utils.utils import print_timestamp



def scatter_3d(coords1, coords2, coords3, cartesian=True):    
    # (x, y, z) or (r, theta, phi)
    
    if not cartesian:
        coords1, coords2, coords3 = sphere2cart(coords1, coords2, coords3)
    
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(coords1, coords2, coords3, depthshade=True)
    plt.show()



def render_coeffs(coeffs, theta_phi_sampling_file, sh_order=5, radius=30):
    
    assert len(coeffs)==(sh_order+1)**2, 'SH order and number of coefficients do not match.'
    
    coeffs = np.array(coeffs)
    coeffs = coeffs/coeffs[0]
    coeffs *= radius
    
    img_shape = np.array(3*(radius*3,))
    
    theta_phi_sampling = np.load(theta_phi_sampling_file)
    h2s = harmonics2sampling(sh_order, theta_phi_sampling)
    r_sampling = h2s.convert(coeffs[np.newaxis,:])
    
    instance_mask = sampling2instance([img_shape//2,], r_sampling, theta_phi_sampling, img_shape, verbose=True)
    
    return instance_mask
        
    

def instance2sampling(instances, theta_phi_sampling, bg_label=1, centroids=None, verbose=True):
    
    # Get labels
    labels = np.unique(instances)
    labels = np.array(list(set(labels)-set([bg_label])))
    
    sampling = np.zeros((len(labels),theta_phi_sampling.shape[0]))
    
    if centroids is None:
        get_centroids = True
        centroids = np.zeros((len(labels),3))
    else:
        get_centroids = False
        assert len(labels) == len(centroids), 'There needs to be a centroid for each label!'

    for num_label, label in enumerate(labels):
        
        if verbose: print_timestamp('Sampling instance {0}/{1}...', [num_label+1, len(labels)])
    
        if np.count_nonzero(instances==label) < 9:
            print_timestamp('Skipping label {0} due to its tiny size.', [label])
            continue
    
        # get inner boundary of the current instance
        instance_inner = morphology.binary_erosion(instances==label, selem=morphology.ball(1))
        instance_inner = np.logical_xor(instance_inner, instances==label)
    
        # get binary instance mask
        x,y,z = np.where(instance_inner)
        if get_centroids:
            centroids[num_label,:] = [x.mean(),y.mean(),z.mean()]
        x -= int(centroids[num_label,0])
        y -= int(centroids[num_label,1])
        z -= int(centroids[num_label,2])
        r,theta,phi = cart2sphere(x, y, z)
        
        # find closest sampling angles
        sampling_tree = cKDTree(np.array([theta,phi]).T)
        _, assignments = sampling_tree.query(theta_phi_sampling, k=3)
        
        # get sampling 
        sampling[num_label,:] = np.mean(r[assignments], axis=1)
        
    return labels, centroids, sampling



def sampling2instance(centroids, sampling, theta_phi_sampling, shape, verbose=True):
    
    instances = np.full(shape, 0, dtype=np.uint16)    
    idx = np.reshape(np.indices(shape), (3,-1))
    
    label = 0
    for centroid, r in zip(centroids, sampling):
        label += 1
        if verbose: print_timestamp('Reconstructing instance {0}/{1}...', [label, sampling.shape[0]])
        
        x,y,z = sphere2cart(r, theta_phi_sampling[:,0], theta_phi_sampling[:,1])
        x += centroid[0]
        y += centroid[1]
        z += centroid[2]
        
        delaunay_tri = Delaunay(np.array([x,y,z]).T)
        
        voxel_idx = delaunay_tri.find_simplex(idx.T).reshape(shape)>0
        
        instances[voxel_idx] = label
        
    return instances




class sampling2harmonics():
    
    def __init__(self, sh_order, theta_phi_sampling, lb_lambda=0.006):
        super(sampling2harmonics, self).__init__()
        self.sh_order = sh_order
        self.theta_phi_sampling = theta_phi_sampling
        self.lb_lambda = lb_lambda
        self.num_samples = len(theta_phi_sampling)
        self.num_coefficients = np.int((self.sh_order+1)**2)
        
        b = np.zeros((self.num_samples, self.num_coefficients))
        l = np.zeros((self.num_coefficients, self.num_coefficients))
        
        for num_sample in range(self.num_samples):
            num_coefficient = 0
            for num_order in range(self.sh_order+1):
                for num_degree in range(-num_order, num_order+1):
                    
                    theta = theta_phi_sampling[num_sample][0]
                    phi = theta_phi_sampling[num_sample][1]
                    
                    y = sph_harm(np.abs(num_degree), num_order, phi, theta)
                                
                    if num_degree < 0:
                        b[num_sample, num_coefficient] = np.real(y) * np.sqrt(2)
                    elif num_degree == 0:
                        b[num_sample, num_coefficient] = np.real(y)
                    elif num_degree > 0:
                        b[num_sample, num_coefficient] = np.imag(y) * np.sqrt(2)
        
                    l[num_coefficient, num_coefficient] = self.lb_lambda * num_order ** 2 * (num_order + 1) ** 2
                    num_coefficient += 1
                    
        b_inv = np.linalg.pinv(np.matmul(b.transpose(), b) + l)
        self.convert_mat = np.matmul(b_inv, b.transpose()).transpose()
        
    def convert(self, r_sampling):
        converted_samples = np.zeros((r_sampling.shape[0],self.num_coefficients))
        for num_sample, r_sample in enumerate(r_sampling):
            r_converted = np.matmul(r_sample[np.newaxis], self.convert_mat)
            converted_samples[num_sample] = np.squeeze(r_converted)
        return converted_samples
            

            
    
class harmonics2sampling():
    
    def __init__(self, sh_order, theta_phi_sampling):
        super(harmonics2sampling, self).__init__()
        self.sh_order = sh_order
        self.theta_phi_sampling = theta_phi_sampling
        self.num_samples = len(theta_phi_sampling)
        self.num_coefficients = np.int((self.sh_order+1)**2)
        
        convert_mat = np.zeros((self.num_coefficients, self.num_samples))
        
        for num_sample in range(self.num_samples):
            num_coefficient = 0
            for num_order in range(self.sh_order+1):
                for num_degree in range(-num_order, num_order+1):
                    
                    theta = theta_phi_sampling[num_sample][0]
                    phi = theta_phi_sampling[num_sample][1]
                    
                    y = sph_harm(np.abs(num_degree), num_order, phi, theta)
                                
                    if num_degree < 0:
                        convert_mat[num_coefficient, num_sample] = np.real(y) * np.sqrt(2)
                    elif num_degree == 0:
                        convert_mat[num_coefficient, num_sample] = np.real(y)
                    elif num_degree > 0:
                        convert_mat[num_coefficient, num_sample] = np.imag(y) * np.sqrt(2)
        
                    num_coefficient += 1
                    
        self.convert_mat = convert_mat
        
    def convert(self, r_harmonic):
        converted_harmonics = np.zeros((r_harmonic.shape[0],self.theta_phi_sampling.shape[0]))
        for num_sample, r_sample in enumerate(r_harmonic):
            r_converted = np.matmul(r_sample[np.newaxis], self.convert_mat)
            converted_harmonics[num_sample] = np.squeeze(r_converted)
        return converted_harmonics
         



def sphere_intersection_poolhelper(instance_indices, point_coords=None, radii=None):
    
    # get radiii, positions and distance
    r1 = radii[instance_indices[0]]
    r2 = radii[instance_indices[1]]
    p1 = point_coords[instance_indices[0]]
    p2 = point_coords[instance_indices[1]]
    d = np.sqrt(np.sum((np.array(p1)-np.array(p2))**2))
    
    # calculate individual volumes
    vol1 = 4/3*np.pi*r1**3
    vol2 = 4/3*np.pi*r2**3
    
    # calculate intersection of volumes
    
    # Smaller sphere inside the bigger sphere
    if d <= np.abs(r1-r2): 
        intersect_vol = 4/3*np.pi*np.minimum(r1,r2)**3
    # No intersection at all
    elif d > r1+r2:
        intersect_vol = 0
    # Partially intersecting spheres
    else:
        intersect_vol = np.pi * (r1 + r2 - d)**2 * (d**2 + 2*d*r2 - 3*r2**2 + 2*d*r1 + 6*r2*r1 - 3*r1**2) / (12*d) 
        
    return (intersect_vol, vol1, vol2)



def harmonic_non_max_suppression(point_coords, point_probs, shape_descriptors, overlap_thresh=0.5, dim_scale=(1,1,1), num_kernel=1, **kwargs):

    if len(point_coords)>3000:
        
        print_timestamp('Too many points, aborting NMS')
        nms_coords = point_coords[:3000]
        nms_probs = point_probs[:3000]
        nms_shapes = shape_descriptors[:3000]
    
    elif len(point_coords)>1:
        
        dim_scale = [d/np.min(dim_scale) for d in dim_scale]
        point_coords_uniform = []
        for point_coord in point_coords:
            point_coords_uniform.append(tuple([p*d for p,d in zip(point_coord,dim_scale)]))
        
        # calculate upper and lower volumes
        r_upper = [r.max() for r in shape_descriptors]
        r_lower = [r.min() for r in shape_descriptors]
        
        # Calculate intersections of lower and upper spheres
        #instance_indices = list(itertools.combinations(range(len(point_coords)), r=2))
        r_max = np.max(r_upper)
        instance_indices = [ (i, j) for i in range(len(point_coords))
                                    for j in range(i+1, len(point_coords))
                                    if np.sum(np.sqrt(np.abs(np.array(point_coords[i])-np.array(point_coords[j])))) < r_max*2 ]
        with mp.Pool(processes=num_kernel) as p:
            vol_upper = p.map(partial(sphere_intersection_poolhelper, point_coords=point_coords_uniform, radii=r_upper), instance_indices)
            vol_lower = p.map(partial(sphere_intersection_poolhelper, point_coords=point_coords_uniform, radii=r_lower), instance_indices)
        
        instances_keep = np.ones((len(point_coords),), dtype=np.bool)
                
        # calculate overlap measure
        for inst_idx, v_up, v_low in zip(instance_indices, vol_upper, vol_lower):
            
            # average intersection with smaller sphere
            overlap_measure_up = v_up[0] / np.minimum(v_up[1],v_up[2])
            overlap_measure_low = v_low[0] / np.minimum(v_low[1],v_low[2])
            overlap_measure = (overlap_measure_up+overlap_measure_low)/2
                        
            if overlap_measure > overlap_thresh:
                # Get min and max probable indice
                inst_min = inst_idx[np.argmin([point_probs[i] for i in inst_idx])]
                inst_max = inst_idx[np.argmax([point_probs[i] for i in inst_idx])]          
                
                # If there already was an instance with higher probability, don't add the current "winner"
                if instances_keep[inst_max] == 0: 
                    # Mark both as excluded
                    instances_keep[inst_min] = 0 
                    instances_keep[inst_max] = 0
                else:                    
                    # Exclude the loser
                    instances_keep[inst_min] = 0 
                    #instances_keep[inst_max] = 1
          
        # Mark remaining indices for keeping
        #instances_keep = instances_keep != -1 
        
        nms_coords = [point_coords[i] for i,v in enumerate(instances_keep) if v]
        nms_probs = [point_probs[i] for i,v in enumerate(instances_keep) if v]
        nms_shapes = [shape_descriptors[i] for i,v in enumerate(instances_keep) if v]
    
    else:
        nms_coords = point_coords
        nms_probs = point_probs
        nms_shapes = shape_descriptors
        
    return nms_coords, nms_shapes, nms_probs


        