# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 15:55:28 2020

@author: Nutzer
"""

import os
import pyshtools
import numpy as np
import gryds

from skimage import io, filters, morphology, measure
from scipy.ndimage import convolve, distance_transform_edt, gaussian_filter, rotate
from pyquaternion import Quaternion
from dipy.core.geometry import sphere2cart

from utils.utils import print_timestamp
from utils.harmonics import harmonics2sampling, sampling2instance
from utils.h5_converter import h5_writer



def generate_data(synthesizer, save_path, experiment_name='dummy_nuclei', num_imgs=50, img_shape=(140,140,1000), max_radius=40, min_radius=20, radius_range=-1, psf=None,\
                  sh_order=20, num_cells=200, num_cells_range=50, circularity=5, smooth_std=0.5, noise_std=0.1, noise_mean=-0.1, position_std=3, z_anisotropy=0.09,\
                  cell_elongation=1.5, irregularity_extend=50, generate_images=False, theta_phi_sampling_file=r'utils/theta_phi_sampling_5000points_10000iter.npy'):
        
    # Set up the synthesizer
    synthesizer = synthesizer(img_shape=img_shape, max_radius=max_radius, min_radius=min_radius,\
                              smooth_std=smooth_std, noise_std=noise_std, noise_mean=noise_mean,\
                              sh_order=sh_order, circularity=circularity, num_cells=num_cells, psf=psf,\
                              position_std=position_std, theta_phi_sampling_file=theta_phi_sampling_file,\
                              cell_elongation=cell_elongation, irregularity_extend=irregularity_extend,
                              generate_images=generate_images, z_anisotropy=z_anisotropy)    
        
    # Set up the save directories
    if generate_images:
        os.makedirs(os.path.join(save_path, 'images'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'masks'), exist_ok=True)
    
    for num_data in range(num_imgs):
        
        if radius_range<0:
            synthesizer.max_radius = max_radius
            synthesizer.min_radius = min_radius 
        else:
            current_radius = np.random.randint(min_radius, max_radius)
            synthesizer.max_radius = current_radius + radius_range
            synthesizer.min_radius = current_radius - radius_range
        
        cell_count = np.random.randint(num_cells-num_cells_range, num_cells+num_cells_range)
        synthesizer.num_cells = cell_count
        
        print_timestamp('_'*20)
        print_timestamp('Generating image {0}/{1} with {2} cells of size {3}-{4}', [num_data+1, num_imgs, cell_count, synthesizer.min_radius, synthesizer.max_radius])
        
        # Get the image and the corresponding mask
        processed_img, instance_mask = synthesizer.generate_data()
        
        ## Save the image
        for num_img,img in enumerate(processed_img):
          
            if not img is None:
                save_name_img = 'psf{0}_img_'.format(num_img)+experiment_name+'_{0}'.format(num_data)  
              
                # TIF
                io.imsave(os.path.join(save_path, 'images', save_name_img+'.tif'), 255*img.astype(np.uint8))
                
                # H5
                img = img.astype(np.float32)
                perc01, perc99 = np.percentile(img, [1,99])
                if not perc99-perc01 <= 0:
                    img -= perc01
                    img /= (perc99-perc01)
                else:
                    img /= img.max()
                img = np.clip(img, 0, 1)
                h5_writer([img], save_name_img+'.h5', group_root='data', group_names=['image'])
                
        ## Save the mask
        save_name_mask = 'mask_'+experiment_name+'_{0}'.format(num_data)
        
        # TIF
        io.imsave(os.path.join(save_path, 'masks', save_name_mask+'.tif'), instance_mask.astype(np.uint16))
    
        # H5
        h5_writer([instance_mask, synthesizer.dist_map], os.path.join(save_path, 'masks', save_name_mask+'.h5'), group_root='data', group_names=['instances', 'distance'])
        
        
        
        


class SyntheticNuclei:
    
    def __init__(self, img_shape=(200,400,400), max_radius=50, min_radius=20, psf=None, sh_order=20, smooth_std=1,\
                 noise_std=0.1, noise_mean=0, num_cells=10, circularity=5, generate_images=False,\
                 theta_phi_sampling_file=r'utils/theta_phi_sampling_5000points_10000iter.npy', **kwargs):
        
        self.img_shape = img_shape
        self.max_radius = max_radius
        self.min_radius = min_radius
        self.sh_order = sh_order
        self.num_coefficients = (sh_order+1)**2
        self.smooth_std = smooth_std
        self.noise_std = noise_std
        self.noise_mean = noise_mean
        self.circularity = circularity
        self.num_cells = num_cells
        self.generate_images = generate_images
        self.theta_phi_sampling_file = theta_phi_sampling_file
        
        if not isinstance(psf, (tuple, list)):
            psf = [psf]
        
        self.psf = []
        for p in psf:
            if isinstance(p, str):
                if psf.endswith(('.tif', '.TIF', 'png')):
                    self.psf.append(io.imread(psf))
                elif psf.endswith(('.npz', '.npy')):
                    self.psf.append(np.load(p))
                else:
                    raise TypeError('Unknown PSF file format.')
            else:
                self.psf.append(p)
         
        self.fg_map = None
        self.instance_mask = None
        self.processed_img = [None]
        
        self._preparations()
        
        
    def _preparations(self):
        # Setting up the converter
        print_timestamp('Loading sampling angles...')
        self.theta_phi_sampling = np.load(self.theta_phi_sampling_file)
        print_timestamp('Setting up harmonic converter...')
        self.h2s = harmonics2sampling(self.sh_order, self.theta_phi_sampling)
        
        
    def generate_data(self, foreground=None, positions=None):
        
        if foreground is None:
            print_timestamp('Generating foreground...')
            self._generate_foreground()
        else:
            self.fg_map = foreground>0
            
        self._generate_distmap()
        
        if positions is None:
            print_timestamp('Determining cell positions...')
            self.positions = self._generate_positions()
        else:
            self.positions = positions
            
        print_timestamp('Starting cell generation...')
        self._generate_instances()
        
        if self.generate_images:
            print_timestamp('Starting synthesis process...')
            self._generate_image()  
            
        print_timestamp('Finished...')
        
        return self.processed_img, self.instance_mask
    
    
    def _generate_foreground(self):
        
        self.fg_map = np.ones(self.img_shape, dtype=bool)
        
        
    def _generate_distmap(self):
        
        # generate foreground distance map
        fg_map = self.fg_map[::4,::4,::4]
        dist_map = distance_transform_edt(fg_map>=1)        
        dist_map = dist_map - distance_transform_edt(fg_map<1)
        dist_map = dist_map.astype(np.float32)
        
        # rescale to original size
        dist_map = np.repeat(dist_map, 4, axis=0)
        dist_map = np.repeat(dist_map, 4, axis=1)
        dist_map = np.repeat(dist_map, 4, axis=2)
        dim_missmatch = np.array(self.fg_map.shape)-np.array(dist_map.shape)
        if dim_missmatch[0]<0: dist_map = dist_map[:dim_missmatch[0],...]
        if dim_missmatch[1]<0: dist_map = dist_map[:,:dim_missmatch[1],:]
        if dim_missmatch[2]<0: dist_map = dist_map[...,:dim_missmatch[2]]
        dist_map = dist_map.astype(np.float32)
        
        self.dist_map = dist_map


    def _generate_positions(self):
        
        positions = np.zeros((self.num_cells, 3), dtype=np.uint16)
        
        # Get map of possible cell locations
        location_map = self.fg_map.copy()
        cell_size_est = (self.min_radius + self.max_radius) // 2
        slicing = tuple(map(slice, [cell_size_est,]*len(self.img_shape), [s-cell_size_est for s in self.img_shape]))
        location_map[slicing] = True
        
        for cell_count in range(self.num_cells):
            
            # Get random centroid
            location = np.array(np.nonzero(location_map))
            location = location[:,np.random.randint(0, location.shape[1])]
            positions[cell_count,:] = location
        
            # Exclude region of current cell from possible future locations
            slicing = tuple(map(slice, list(np.maximum(location-cell_size_est, 0)), list(location+cell_size_est)))
            location_map[slicing] = False      
                
        return positions
            
    
    def _generate_instances(self):
        
        assert self.circularity>=0, 'Circularity needs to be positive.'
                
        # Get the power per harmonic order
        power_per_order = np.arange(self.sh_order+1, dtype=np.float32)
        power_per_order[0] = np.inf
        power_per_order = power_per_order**-self.circularity
        
        coeff_list = np.zeros((len(self.positions), self.num_coefficients), dtype=np.float32)
        
        for cell_count in range(len(self.positions)):
                        
            # Get harmonic coefficients
            clm = pyshtools.SHCoeffs.from_random(power_per_order)
            coeffs = clm.coeffs
            coeffs[0,0,0] = 1
            
            # Get radius
            radius = np.random.randint(self.min_radius, self.max_radius)
            
            # Scale coefficients respectively
            coeffs *= radius
            coeffs = np.concatenate((np.fliplr(coeffs[0,...]), coeffs[1,...]), axis=1)
            coeffs = coeffs[np.nonzero(coeffs)]
            
            assert len(coeffs) == self.num_coefficients, 'Number of coefficients did not match the expected value.'
            
            coeff_list[cell_count,:] = coeffs
            
           
        # Reconstruct the sampling from the coefficients
        r_sampling = self.h2s.convert(coeff_list)
        
        # Reconstruct the intance mask
        instance_mask = sampling2instance(self.positions, r_sampling, self.theta_phi_sampling, self.img_shape, verbose=True)
        
        self.instance_mask = instance_mask
            
        
        
    def _generate_image(self):
        
        assert not self.instance_mask is None, 'There needs to be an instance mask.'
        
        # Generate image
        img_raw = np.zeros_like(self.instance_mask, dtype=np.float32)
        for label in np.unique(self.instance_mask):
            if label == 0: continue # exclude background
            img_raw[self.instance_mask == label] = np.random.uniform(0.5, 0.9)
            
        self.processed_img  = []
            
        for num_psf,psf in enumerate(self.psf):
            
            print_timestamp('Applying PSF {0}/{1}...', [num_psf+1, len(self.psf)])
        
            img = img_raw.copy()
            
            # Perform PSF smoothing
            if not psf is None:
                img = convolve(img, psf)
            
                # Add final additive noise
                noise = np.random.normal(self.noise_mean, self.noise_std, size=self.img_shape)
                img = img+noise
                img = img.clip(0, 1)
            
            # Final smoothing touch
            img = filters.gaussian(img, self.smooth_std)
            
            self.processed_img.append(img.astype(np.float32))
        
        
        
    
        
    
    
    
#%% 3D CTC Data Sets  
    
class SyntheticCE(SyntheticNuclei):
    
    def __init__(self, img_shape=(35, 512, 708), z_anisotropy=0.09, cell_elongation=1.0, num_cells=4, **kwargs):
        
        super().__init__(img_shape=img_shape, num_cells=num_cells, min_radius=0, max_radius=0)
        self.z_anisotropy = z_anisotropy
        self.cell_elongation = cell_elongation

        
    def _preparations(self):
        pass
    
        
    def _generate_foreground(self):
        
        # Determine positions
        coords = np.indices(self.img_shape, dtype=np.float16)
        coords[0,...] -= self.img_shape[0]//2
        coords[1,...] -= self.img_shape[1]//2
        coords[2,...] -= self.img_shape[2]//2
                        
        # determine ellipsoid parameters (adjusted to the image size)
        a,b,c = [int(i*0.3) for i in self.img_shape]
        
        # within ellipsoid equation: (x/a)^2 + (y/b)^2 + /z/c)^2 < 1
        ellipsoid = (coords[0,...]/a)**2 + (coords[1,...]/b)**2 + (coords[2,...]/c)**2
        self.fg_map = ellipsoid<=1
        
        # determine cell radii (self.num_cells cells should easily fit into the fg region)
        cell_vol_approx = np.count_nonzero(self.fg_map)/(self.num_cells*3)
        r_approx = int((3/(4*np.pi)*cell_vol_approx/self.z_anisotropy)**(1/3))
        self.min_radius = r_approx-1
        self.max_radius = r_approx+1
        
        
        
    def _generate_positions(self):
        
        positions = np.zeros((self.num_cells, 3), dtype=np.uint16)
        
        # Get map of possible cell locations (remove outer shape to keep cells inside)
        locations = np.array(np.nonzero(self.fg_map))
        
        for cell_count in range(self.num_cells):
            
            print_timestamp('Placing cell {0}/{1}...', [cell_count+1, self.num_cells])
            
            # Get cell parameters (2* to remove locations around cell for future centroids)
            cell_shape = np.array([2*self.max_radius*self.z_anisotropy, 2*self.max_radius, 2*self.max_radius])
                
            # Get random centroid
            if locations.shape[1] == 0:
                print_timestamp('The maximum number of cells ({0}) was reached...', [cell_count+1])
                positions = positions[:cell_count-1,:]
                break
                        
            location = locations[:,np.random.randint(0, locations.shape[1])]
            positions[cell_count,:] = location
            
            # Exclude region of current cell from possible future locations
            distances = locations - location[:,np.newaxis]
            distances = distances / cell_shape[:,np.newaxis]
            distances = np.sqrt(np.sum(distances**2, axis=0))
            locations = locations[:,distances>1]
                
        return positions
    
    
    
    def _generate_instances(self):
        
        # create local coordinates
        cell_mask_shape = (int(self.max_radius*2*self.z_anisotropy), self.max_radius*3, self.max_radius*3)
        coords_default = np.indices(cell_mask_shape)
        for i in range(coords_default.shape[0]):
            coords_default[i,...] -= np.max(coords_default[i,...])//2
        
        # place a cell at each position
        instance_mask = np.zeros(self.dist_map.shape, dtype=np.uint16)
        for num_cell, pos in enumerate(self.positions):
            
            print_timestamp('Generating cell {0}/{1}...', [num_cell+1, len(self.positions)])
            
            # For a small fraction of cells, simulate elongation after mitosis
            if np.random.random() < 0.1:
                cell_elongation = 1+np.random.random()*self.cell_elongation
            else:
                cell_elongation = 1+np.random.random()*self.cell_elongation/10
            
            cell_size = np.random.randint(self.min_radius,self.max_radius)
            a,b,c = [cell_size*self.z_anisotropy, cell_size, cell_size/cell_elongation]  
            coords = coords_default.copy()
            
            x_new = coords[0,...]
            y_new = coords[1,...]
            z_new = coords[2,...]
                    
            ellipsoid = ((x_new/a)**2 + (y_new/b)**2 + (z_new/c)**2) <= 1
            
            # Randomly rotate cell
            rot_angle = np.random.randint(-20,20)
            ellipsoid = rotate(ellipsoid, rot_angle, axes=(1,2), reshape=False, order=0)
                        
            slice_start = [np.minimum(np.maximum(0,p-c//2),i-c) for p,c,i in zip(pos,cell_mask_shape,self.img_shape)]
            slice_end = [s+c for s,c in zip(slice_start,cell_mask_shape)]
            slicing = tuple(map(slice, slice_start, slice_end))
            instance_mask[slicing] = np.maximum(instance_mask[slicing], (num_cell+1)*ellipsoid.astype(np.uint16))
            
        self.instance_mask = instance_mask.astype(np.uint16)
        
        
      
        
        
class SyntheticTRIC(SyntheticNuclei):
    
    def __init__(self, img_shape=(13, 2450, 1700), cell_elongation=0.0, min_radius=5, max_radius=6, num_cells=3000, **kwargs):
        
        super().__init__(img_shape=img_shape, num_cells=num_cells, min_radius=min_radius, max_radius=max_radius)
        
        self.cell_elongation = cell_elongation
        
    def _preparations(self):
        pass
      
    
    def _generate_foreground(self):
        
        self.fg_shape = (500,500,500)
        self.coords_fg = np.indices(self.fg_shape)
        for i in range(self.coords_fg.shape[0]):
            self.coords_fg[i,...] -= np.max(self.coords_fg[i,...])//2
            
        self.radius_outside = 230
        self.radius_inside = self.radius_outside - 1
        
        # Create ellipsoids
        a,b,c = 1-0.05*np.random.random(3)        
        ellipsoid_outside = ((self.coords_fg[0,...]/self.radius_outside/a)**2 + (self.coords_fg[1,...]/self.radius_outside/b)**2 + (self.coords_fg[2,...]/self.radius_outside/c)**2) <= 1
        ellipsoid_inside = ((self.coords_fg[0,...]/self.radius_inside/a)**2 + (self.coords_fg[1,...]/self.radius_inside/b)**2 + (self.coords_fg[2,...]/self.radius_inside/c)**2) <= 1
        
        # Randomly rotate 
        rot_angle = np.random.randint(0,360)
        ellipsoid_outside = rotate(ellipsoid_outside, rot_angle, axes=(1,2), reshape=False, order=0)
        ellipsoid_inside = rotate(ellipsoid_inside, rot_angle, axes=(1,2), reshape=False, order=0)
             
        self.fg_map = np.logical_xor(ellipsoid_outside, ellipsoid_inside)
        
        
        
    def _generate_positions(self):
        
        positions = np.zeros((self.num_cells, 3), dtype=np.uint16)
        
        # Get map of possible cell locations (remove outer shape to keep cells inside)
        locations = np.array(np.nonzero(self.fg_map))
        
        for cell_count in range(self.num_cells):
            
            print_timestamp('Placing cell {0}/{1}...', [cell_count+1, self.num_cells])
            
            # Get cell parameters (2* to remove locations around cell for future centroids)
            cell_shape = np.array([2*self.max_radius, 2*self.max_radius, 2*self.max_radius])
                
            # Get random centroid
            if locations.shape[1] == 0:
                print_timestamp('The maximum number of cells ({0}) was reached...', [cell_count+1])
                positions = positions[:cell_count-1,:]
                break
                        
            location = locations[:,np.random.randint(0, locations.shape[1])]
            positions[cell_count,:] = location
            
            # Exclude region of current cell from possible future locations
            distances = locations - location[:,np.newaxis]
            distances = distances / cell_shape[:,np.newaxis]
            distances = np.sqrt(np.sum(distances**2, axis=0))
            locations = locations[:,distances>1]
            
        return positions
    
    
    
    def _generate_instances(self):
        
        # create local coordinates
        cell_mask_shape = (self.max_radius*3, self.max_radius*3, self.max_radius*3)
        coords_default = np.indices(cell_mask_shape)
        for i in range(coords_default.shape[0]):
            coords_default[i,...] -= np.max(coords_default[i,...])//2
        
        # place a cell at each position
        instance_mask = np.zeros(self.fg_shape, dtype=np.uint16)
        for num_cell, pos in enumerate(self.positions):
            
            print_timestamp('Generating cell {0}/{1}...', [num_cell+1, len(self.positions)])
                        
            cell_size = np.random.randint(self.min_radius,self.max_radius)
            cell_elongation = 1+np.random.random()*self.cell_elongation
            a,b,c = [cell_size, cell_size/cell_elongation, cell_size]  
            coords = coords_default.copy()
            
            x_new = coords[0,...]
            y_new = coords[1,...]
            z_new = coords[2,...]
                    
            ellipsoid = ((x_new/a)**2 + (y_new/b)**2 + (z_new/c)**2) <= 1
            
            # Randomly rotate cell
            rot_angle = np.random.randint(0,360)
            ellipsoid = rotate(ellipsoid, rot_angle, axes=(1,2), reshape=False, order=0)
                        
            slice_start = [np.minimum(np.maximum(0,p-c//2),i-c) for p,c,i in zip(pos,cell_mask_shape,self.fg_map.shape)]
            slice_end = [s+c for s,c in zip(slice_start,cell_mask_shape)]
            slicing = tuple(map(slice, slice_start, slice_end))
            instance_mask[slicing] = np.maximum(instance_mask[slicing], (num_cell+1)*ellipsoid.astype(np.uint16))
            
        # Get coords of whole instance image
        coords = np.indices(self.fg_shape)
        for i in range(coords.shape[0]):
            coords[i,...] -= np.max(coords[i,...])//2
            
        indices = np.indices(self.img_shape)
        
        # Reverse planar projection (Map to Globe)
        r = indices[0,...]+self.radius_inside-self.img_shape[0]//2
        phi = (indices[1,...]/self.img_shape[1])*2*np.pi - np.pi
        theta = (indices[2,...]/self.img_shape[2])*np.pi*0.9 +  np.pi*0.05 # exclude overly stretched cells at the image border/poles
        
        # Position in original image
        z_ell,y_ell,x_ell = sphere2cart(r,theta,phi)
        z_ell += self.fg_shape[0]/2
        y_ell += self.fg_shape[1]/2
        x_ell += self.fg_shape[2]/2
        
        z_ell = z_ell.astype(int)
        y_ell = y_ell.astype(int)
        x_ell = x_ell.astype(int)
        
        # Fill in value
        mercator_map= instance_mask[z_ell,y_ell,x_ell]
            
        self.instance_mask = mercator_map.astype(np.uint16)
        
     

#%% 2D CTC Data Sets  
    
class Synthetic2DGOWT1(SyntheticNuclei):
    
    def __init__(self, img_shape=(1,1024,1024), min_radius=30, max_radius=45, cell_elongation=0.3, num_cells=30, **kwargs):
        
        super().__init__(img_shape=img_shape, max_radius=max_radius, min_radius=min_radius, num_cells=num_cells)
        self.cell_elongation = cell_elongation
        
        
    def _preparations(self):
        pass
    
        
        
    def _generate_positions(self):
        
        positions = np.zeros((self.num_cells, 3), dtype=np.uint16)
        
        # Get map of possible cell locations
        locations = np.array(np.nonzero(self.fg_map))
        
        
        for cell_count in range(self.num_cells):
            
            print_timestamp('Placing cell {0}/{1}...', [cell_count+1, self.num_cells])
            
            # Get random centroid
            if locations.shape[1] == 0:
                print_timestamp('The maximum number of cells ({0}) was reached...', [cell_count+1])
                positions = positions[:cell_count-1,:]
                break
                        
            location = locations[:,np.random.randint(0, locations.shape[1])]
            positions[cell_count,:] = location
            
            # Exclude region of current cell from possible future locations
            distances = locations[1:,:] - location[1:,np.newaxis]
            distances = distances / (2*self.max_radius)
            distances = np.sqrt(np.sum(distances**2, axis=0))
            locations = locations[:,distances>1]
                
        return positions
    
    
    
    def _generate_instances(self):
                       
        # create local coordinates
        cell_mask_shape = (1,self.max_radius*3,self.max_radius*3)
        coords_default = np.indices(cell_mask_shape)
        for i in range(coords_default.shape[0]):
            coords_default[i,...] -= np.max(coords_default[i,...])//2
        
        # place a cell at each position
        instance_mask = np.zeros(self.dist_map.shape, dtype=np.uint16)
        for num_cell, pos in enumerate(self.positions):
            
            print_timestamp('Generating cell {0}/{1}...', [num_cell+1, len(self.positions)])
            
            # Get random cell layout
            cell_size = np.random.randint(self.min_radius,self.max_radius)
            cell_elongation = 1+np.random.random()*self.cell_elongation
            a,b,c = [1, cell_size/cell_elongation, cell_size]  
            coords = coords_default.copy()
            
            x_new = coords[0,...]
            y_new = coords[1,...]
            z_new = coords[2,...]
                    
            ellipsoid = ((x_new/a)**2 + (y_new/b)**2 + (z_new/c)**2) <= 1
            
            # Randomly rotate cell
            rot_angle = np.random.randint(0,360)
            ellipsoid = rotate(ellipsoid, rot_angle, axes=(1,2), reshape=False, order=0)
            
            # Randomly deform cell
            random_grid = np.random.rand(2,3,3)
            random_grid -= 0.5
            random_grid /= 10        
            bspline = gryds.BSplineTransformation(random_grid)
            interpolator = gryds.Interpolator(ellipsoid[0,...], order=0)
            ellipsoid[0,...] = interpolator.transform(bspline)
            ellipsoid = np.clip(ellipsoid, 0, 1)
            
            # Add cell to image
            slice_start = [np.minimum(np.maximum(0,p-c//2),i-c) for p,c,i in zip(pos,cell_mask_shape,self.img_shape)]
            slice_end = [s+c for s,c in zip(slice_start,cell_mask_shape)]
            slicing = tuple(map(slice, slice_start, slice_end))
            instance_mask[slicing] = np.maximum(instance_mask[slicing], (num_cell+1)*ellipsoid.astype(np.uint16))
            
        self.instance_mask = instance_mask.astype(np.uint16)
            
        
        
        
  
class Synthetic2DHeLa(SyntheticNuclei):
    
    def __init__(self, img_shape=(1,700,1100), min_radius=10, max_radius=20, cell_elongation=1.0, num_cells=450, **kwargs):
        
        super().__init__(img_shape=img_shape, max_radius=max_radius, min_radius=min_radius, num_cells=num_cells)
        self.cell_elongation = cell_elongation
        
        
        
    def _preparations(self):
        pass
    
    
    
    def _generate_foreground(self):
        
        self.fg_map = np.zeros(self.img_shape, dtype=bool)
        locations_all = np.indices(self.fg_map.shape)
        
        # add locations for clusters of cells
        num_clusters = np.random.randint(15,30)
        locations = locations_all.copy()
        locations = locations.reshape((3,-1))
        locations = locations[:,np.random.randint(locations.shape[1],size=num_clusters)]
        
        for loc in locations.T:
            
                ellipsoid = (((locations_all[0,...]-loc[0])/100)**2 +\
                             ((locations_all[1,...]-loc[1])/100)**2 +\
                             ((locations_all[2,...]-loc[2])/100)**2) <= 1
                
                self.fg_map = np.logical_or(self.fg_map, ellipsoid)
        


    def _generate_positions(self):
        
        positions = np.zeros((self.num_cells, 3), dtype=np.uint16)
        
        # Get map of possible cell locations
        locations = np.array(np.nonzero(self.fg_map))
        
        
        for cell_count in range(self.num_cells):
            
            print_timestamp('Placing cell {0}/{1}...', [cell_count+1, self.num_cells])
            
            # Get random centroid
            if locations.shape[1] == 0:
                print_timestamp('The maximum number of cells ({0}) was reached...', [cell_count+1])
                positions = positions[:cell_count-1,:]
                break
                        
            location = locations[:,np.random.randint(0, locations.shape[1])]
            positions[cell_count,:] = location
            
            # Exclude region of current cell from possible future locations
            distances = locations[1:,:] - location[1:,np.newaxis]
            distances = distances / (2*self.max_radius)
            distances = np.sqrt(np.sum(distances**2, axis=0))
            locations = locations[:,distances>1]
                
        return positions
    
    
    
    def _generate_instances(self):
                       
        # create local coordinates
        cell_mask_shape = (1,self.max_radius*3,self.max_radius*3)
        coords_default = np.indices(cell_mask_shape)
        for i in range(coords_default.shape[0]):
            coords_default[i,...] -= np.max(coords_default[i,...])//2
        
        # place a cell at each position
        instance_mask = np.zeros(self.dist_map.shape, dtype=np.uint16)
        for num_cell, pos in enumerate(self.positions):
            
            print_timestamp('Generating cell {0}/{1}...', [num_cell+1, len(self.positions)])
            
            # Get random cell layout
            cell_size = np.random.randint(self.min_radius,self.max_radius)
            cell_elongation = 1 + np.random.random()*self.cell_elongation * (self.min_radius/cell_size)
            a,b,c = [1, cell_size/cell_elongation, cell_size]  
            coords = coords_default.copy()
            
            x_new = coords[0,...]
            y_new = coords[1,...]
            z_new = coords[2,...]
                    
            ellipsoid = ((x_new/a)**2 + (y_new/b)**2 + (z_new/c)**2) <= 1
            
            # Randomly rotate cell
            rot_angle = np.random.randint(0,360)
            ellipsoid = rotate(ellipsoid, rot_angle, axes=(1,2), reshape=False, order=0)
            
            # Randomly deform cell
            random_grid = np.random.rand(2,3,3)
            random_grid -= 0.5
            random_grid /= 5        
            bspline = gryds.BSplineTransformation(random_grid)
            interpolator = gryds.Interpolator(ellipsoid[0,...], order=0)
            ellipsoid[0,...] = interpolator.transform(bspline)
            ellipsoid = np.clip(ellipsoid, 0, 1)
            
            # Add cell to image
            slice_start = [np.minimum(np.maximum(0,p-c//2),i-c) for p,c,i in zip(pos,cell_mask_shape,self.img_shape)]
            slice_end = [s+c for s,c in zip(slice_start,cell_mask_shape)]
            slicing = tuple(map(slice, slice_start, slice_end))
            instance_mask[slicing] = np.maximum(instance_mask[slicing], (num_cell+1)*ellipsoid.astype(np.uint16))
            
        self.instance_mask = instance_mask.astype(np.uint16)
            
        
        