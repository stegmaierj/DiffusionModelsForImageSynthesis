# -*- coding: utf-8 -*-


import os
import h5py
import csv
import itertools
import numpy as np

from scipy.ndimage import filters, distance_transform_edt
from skimage import io
from torch.utils.data import Dataset

from dataloader.augmenter import intensity_augmenter, geometry_augmenter


class MeristemH5Dataset(Dataset):
    """
    Dataset of fluorescently labeled cell membranes
    """
    
    def __init__(self, list_path, data_root='', patch_size=(64,128,128), data_norm='percentile', shuffle=True, samples_per_epoch=-1,\
                 image_groups=('data/image',), mask_groups=('data/distance', 'data/seeds', 'data/boundary'), patches_from_fg=0.0,\
                 image_noise_channel=-2, mask_noise_channel=-2, noise_type='gaussian',\
                 dist_handling='bool', dist_scaling=(100,100), seed_handling='float', boundary_handling='bool', instance_handling='bool',\
                 correspondence=True, no_img=False, no_mask=False, reduce_dim=False, augmentation_dict=None, **kwargs):
        
        
        # Sanity checks
        assert len(patch_size)==3, 'Patch size must be 3-dimensional.'
        
        if reduce_dim:
            assert np.any([p==1 for p in patch_size]), 'Reduce is only possible, if there is a singleton patch dimension.'
        
        # Save parameters
        self.data_root = data_root
        self.list_path = list_path 
        self.patch_size = tuple(patch_size)
        self.data_norm = data_norm
        self.patches_from_fg = patches_from_fg
        self.dist_handling = dist_handling
        self.dist_scaling = dist_scaling
        self.seed_handling = seed_handling
        self.instance_handling = instance_handling
        self.boundary_handling = boundary_handling
        self.correspondence = correspondence
        self.no_img = no_img
        self.no_mask = no_mask
        self.reduce_dim = reduce_dim
        self.samples_per_epoch = samples_per_epoch
        
        # Read the filelist and construct full paths to each file
        self.shuffle = shuffle
        self.image_groups = image_groups
        self.mask_groups = mask_groups
        self.image_noise_channel = image_noise_channel
        self.mask_noise_channel = mask_noise_channel
        self.noise_type = noise_type
        self.norm1 = 0
        self.norm2 = 1
        self.data_list = self._read_list()
        
        if self.samples_per_epoch < len(self.data_list) and self.samples_per_epoch>0:
            print('Only {0}/{1} files are used in each epoch! Consider increasing the number of samples per epoch.'.format(self.samples_per_epoch, len(self.data_list)))
    
        # Get image statistics from up to 10 files
        if not self.no_img and 'image' in self.image_groups[0]:
            print('Getting statistics from images...')
            self.data_statistics = {'min':[], 'max':[], 'mean':[], 'std':[], 'perc01':[], 'perc99':[]}
            for file_pair in self.data_list[:20]:
                with h5py.File(file_pair[0], 'r') as f_handle:
                    image = f_handle[self.image_groups[0]][...].astype(np.float32)
                    self.data_statistics['min'].append(np.min(image))
                    self.data_statistics['max'].append(np.max(image))
                    self.data_statistics['mean'].append(np.mean(image))
                    self.data_statistics['std'].append(np.std(image))
                    perc01, perc99 = np.percentile(image, [1,99])
                    self.data_statistics['perc01'].append(perc01)
                    self.data_statistics['perc99'].append(perc99)
            
            # Construct data set statistics
            self.data_statistics['min'] = np.min(self.data_statistics['min'])
            self.data_statistics['max'] = np.max(self.data_statistics['max'])
            self.data_statistics['mean'] = np.mean(self.data_statistics['mean'])
            self.data_statistics['std'] = np.mean(self.data_statistics['std'])
            self.data_statistics['perc01'] = np.mean(self.data_statistics['perc01'])
            self.data_statistics['perc99'] = np.mean(self.data_statistics['perc99'])
            
            # Get the normalization values
            if self.data_norm == 'minmax':
                self.norm1 = self.data_statistics['min']
                self.norm2 = self.data_statistics['max']-self.data_statistics['min']
            elif self.data_norm == 'minmax_shifted':
                self.norm1 = 0.5*(self.data_statistics['max']+self.data_statistics['min'])
                self.norm2 = 0.5*(self.data_statistics['max']-self.data_statistics['min'])
            elif self.data_norm == 'meanstd':
                self.norm1 = self.data_statistics['mean']
                self.norm2 = self.data_statistics['std']
            elif self.data_norm == 'percentile':
                self.norm1 = self.data_statistics['perc01']
                self.norm2 = self.data_statistics['perc99']-self.data_statistics['perc01']
            
        # Construct the augmentation dict
        if augmentation_dict is None:
            self.augmentation_dict = {}
        else:
            self.augmentation_dict = augmentation_dict
                
        self.intensity_augmenter = intensity_augmenter(self.augmentation_dict)
        self.augmentation_dict = self.intensity_augmenter.augmentation_dict
        self.geometry_augmenter = geometry_augmenter(self.augmentation_dict)
        self.augmentation_dict = self.geometry_augmenter.augmentation_dict
        self.permute_dim = self.augmentation_dict['permute_dim'] if 'permute_dim' in self.augmentation_dict.keys() else False
        
        
    def test(self, test_folder='', num_files=10):
        
        os.makedirs(test_folder, exist_ok=True)
        
        for i in range(num_files):
            test_sample = self.__getitem__(i%self.__len__())   
            
            if not self.no_img:
                for num_img in range(test_sample['image'].shape[0]):
                    io.imsave(os.path.join(test_folder, 'train_img{0}_group{1}.tif'.format(i,num_img)), test_sample['image'][num_img,...], check_contrast=False)
            
            if not self.no_mask:
                for num_mask in range(test_sample['mask'].shape[0]):
                    io.imsave(os.path.join(test_folder, 'train_mask{0}_group{1}.tif'.format(i,num_mask)), test_sample['mask'][num_mask,...], check_contrast=False)
        
        
    def _read_list(self):
        
        # Read the filelist and create full paths to each file
        filelist = []    
        with open(self.list_path, 'r') as f:
            reader = csv.reader(f, delimiter=';')
            for row in reader:
                if len(row)==0 or np.sum([len(r) for r in row])==0: continue
                row = [os.path.join(self.data_root, r) for r in row]
                filelist.append(row)
        
        if self.shuffle:
            np.random.shuffle(filelist)
                
        return filelist
    
    
    def __len__(self):
        
        if self.samples_per_epoch<1:
            return len(self.data_list)
        else:
            return self.samples_per_epoch
    
    
    def _normalize(self, data, group_name):
        
        # Normalization
            
        if 'distance' in group_name:
            if self.dist_handling == 'float':
                data /= self.dist_scaling[0]
            elif self.dist_handling == 'bool':
                data = data<0
            elif self.dist_handling == 'bool_inv':
                data = data>=0
            elif self.dist_handling == 'exp':
                data = (data/self.dist_scaling[0])**3
            elif self.dist_handling == 'tanh':    
                foreground = np.float16(data>0)
                data = np.tanh(data/self.dist_scaling[0])*foreground + np.tanh(data/self.dist_scaling[1])*(1-foreground)
            else:
                pass
            
        elif 'seed' in group_name:                    
            if self.seed_handling == 'float':
                data = data.astype(np.float32)
                data = filters.gaussian_filter(data, 2)
                if np.max(data)>1e-4: data /= float(np.max(data))
            elif self.seed_handling == 'bool':
                data = data>0.1
            else:
                pass
         
        elif 'instance' in group_name or 'nuclei' in group_name:
            if self.instance_handling == 'bool':
                data = data>0
         
        elif 'boundary' in group_name:
            if self.boundary_handling == 'bool':
                data = data>0
                
        elif 'image' in group_name:
            data -= self.norm1
            data /= self.norm2
            if self.data_norm == 'minmax' or self.data_norm == 'percentile':
                data = np.clip(data, 1e-5, 1)
            elif self.data_norm == 'minmax_shifted':
                data = np.clip(data, -1, 1)
                        
        return data
    
    
    def _get_noise(self, shape):
        
        if self.noise_type.lower() == 'gaussian':
            noise = np.random.randn(*shape)
        elif self.noise_type.lower() == 'rayleigh':
            noise = np.random.rayleigh(size=shape)
        elif self.noise_type.lower() == 'laplace':
            noise = np.random.laplace(size=shape)
        else:
            raise ValueError('Unknown noise distribution "{0}". Choose from (gaussian,rayleigh,laplace)'.format(self.noise_type))
        
        return noise.astype(np.float32)
        
    
    def __getitem__(self, idx):
        
        idx = idx%len(self.data_list)
        
        # Get the paths to the image and mask
        filepath = self.data_list[idx]
        
        # Permute patch dimensions
        patch_size = list(self.patch_size)
        if self.permute_dim:
            swaps = np.random.choice(range(3), 2, replace=False)
            patch_size[swaps[0]], patch_size[swaps[1]] = patch_size[swaps[1]], patch_size[swaps[0]]
            
        sample = {}
        
        if not self.no_mask:
            
            # Load the mask patch
            mask = np.zeros((len(self.mask_groups),)+self.patch_size, dtype=np.float32)
            with h5py.File(filepath[1], 'r') as f_handle:
                for num_group, group_name in enumerate(self.mask_groups):
                    
                    mask_tmp = f_handle[group_name]     
                    
                    if np.any(np.isnan(mask_tmp)): print('mask')
                    
                    # determine the patch position for the first mask
                    if num_group==0:
                        
                        # obtain patch position from foreground indices or random
                        if self.patches_from_fg > np.random.random():
                            fg_indices = np.where(mask_tmp)
                            rnd_start = [np.maximum(0,f[np.random.randint(len(fg_indices[0]))]-p) for f,p in zip(fg_indices,patch_size)]
                        else:
                            rnd_start = [np.random.randint(0, np.maximum(1,mask_dim-patch_dim)) for patch_dim, mask_dim in zip(patch_size, mask_tmp.shape)]
                        rnd_end = [start+patch_dim for start, patch_dim in zip(rnd_start, patch_size)]    
                        slicing = tuple(map(slice, rnd_start, rnd_end))    
                        
                    # extract the patch
                    mask_tmp = mask_tmp[slicing].astype(np.float32)
                    
                    # Pad if neccessary
                    pad_width = [(0,np.maximum(0,p-i)) for p,i in zip(patch_size,mask_tmp.shape)] 
                    mask_tmp = np.pad(mask_tmp, pad_width, mode='reflect')     
                    
                    # Permute dimensions
                    if self.permute_dim:
                        mask_tmp = np.swapaxes(mask_tmp, *swaps)
                        # sanity check
                        assert mask_tmp.shape == tuple(self.patch_size), 'Mask dimension missmatch after rotation. {0} instead of {1}.'.format(mask_tmp.shape, self.patch_size)
                        
                    # Apply geometry augmentations 
                    mask_tmp = self.geometry_augmenter.apply(mask_tmp, reset=num_group==0)     
                    if 'flow' in group_name and self.permute_dim:
                        num_group = num_group if not num_group-1 in swaps else int(swaps[swaps!=num_group-1]+1)
                   
                    # Normalization
                    mask_tmp = self._normalize(mask_tmp, group_name)
                    mask_tmp = mask_tmp.astype(np.float32)
                    
                    # Add noise to current channel, if desired
                    if self.mask_noise_channel == num_group:
                        noise = self._get_noise(mask_tmp.shape)
                        mask_tmp += noise
                    
                    # Store current mask
                    mask[num_group,...] = mask_tmp
                        
            mask = mask.astype(np.float32)
            
            if self.reduce_dim:
                out_shape = [p for i,p in enumerate(mask.shape) if p!=1 or i==0]
                mask = np.reshape(mask, out_shape)
                    
            if self.mask_noise_channel == -1:
                noise = self._get_noise((1,)+mask.shape[1:])
                mask = np.concatenate((mask, noise), axis=0)
                
            sample['mask'] = mask
                
                
        if not self.no_img:
            
            # Load the image patch
            image = np.zeros((len(self.image_groups),)+self.patch_size, dtype=np.float32)
            with h5py.File(filepath[0], 'r') as f_handle:
                for num_group, group_name in enumerate(self.image_groups):
                    
                    image_tmp = f_handle[group_name]   
                    
                    # Check if positioning and geometrical augmentations have to be reset
                    reset = (self.no_mask or not self.correspondence) and num_group==0
                                
                    # Determine the patch position
                    if reset:
                        rnd_start = [np.random.randint(0, np.maximum(1,image_dim-patch_dim)) for patch_dim, image_dim in zip(patch_size, image_tmp.shape)]
                        rnd_end = [start+patch_dim for start, patch_dim in zip(rnd_start, patch_size)]    
                        slicing = tuple(map(slice, rnd_start, rnd_end))
                        
                    # extract the patch
                    image_tmp = image_tmp[slicing].astype(np.float32)
                    
                    # Pad if neccessary
                    pad_width = [(0,np.maximum(0,p-i)) for p,i in zip(patch_size,image_tmp.shape)] 
                    image_tmp = np.pad(image_tmp, pad_width, mode='reflect')
                    
                    # Permute dimensions
                    if self.permute_dim:
                        image_tmp = np.swapaxes(image_tmp, *swaps)
                        # sanity check
                        assert image_tmp.shape == tuple(self.patch_size), 'Image dimension missmatch after rotation. {0} instead of {1}.'.format(image.shape, self.patch_size)
                        
                    # Apply geometry augmentation
                    image_tmp = self.geometry_augmenter.apply(image_tmp, reset=reset)

                    # Normalization
                    image_tmp = self._normalize(image_tmp, group_name)

                    # Apply intensity augmentations
                    if 'image' in group_name:    
                        image_tmp = self.intensity_augmenter.apply(image_tmp)
                        
                    # Add noise to current channel, if desired
                    if self.image_noise_channel == num_group:
                        if self.correspondence and self.mask_noise_channel==num_group and not self.no_mask:
                            pass # reuse the noise channel from the corresponding mask
                        else:
                            noise = self._get_noise(image_tmp.shape)
                        image_tmp += noise
                    
                    image[num_group,...] = image_tmp
                        
            image = image.astype(np.float32)
                    
            if self.reduce_dim:
                out_shape = [p for i,p in enumerate(image.shape) if p!=1 or i==0]
                image = np.reshape(image, out_shape)
                    
            if self.image_noise_channel == -1:
                if self.correspondence and self.mask_noise_channel==-1 and not self.no_mask:
                    pass # reuse the noise channel from the corresponding mask
                else:
                    noise = self._get_noise((1,)+image.shape[1:])
                image = np.concatenate((image, noise), axis=0)
                
            sample['image'] =  image
                
        return sample
    

    
    
    
    
class MeristemH5Tiler(Dataset):
    
    """
    Dataset of fluorescently labeled cell membranes
    """
    
    def __init__(self, list_path, data_root='', patch_size=(64,128,128), overlap=(10,10,10), crop=(10,10,10), data_norm='percentile',\
                 image_groups=('data/image',), mask_groups=('data/distance', 'data/seeds', 'data/boundary'),\
                 image_noise_channel=-2, mask_noise_channel=-2, noise_type='gaussian',\
                 dist_handling='bool', dist_scaling=(100,100), seed_handling='float', boundary_handling='bool', instance_handling='bool',\
                 no_mask=False, no_img=False, reduce_dim=False, **kwargs):
           
        # Sanity checks
        assert len(patch_size)==3, 'Patch size must be 3-dimensional.'
        
        if reduce_dim:
            assert np.any([p==1 for p in patch_size]), 'Reduce is only possible, if there is a singleton patch dimension.'
        
        # Save parameters
        self.data_root = data_root
        self.list_path = os.path.abspath(list_path) 
        self.patch_size = tuple(patch_size)
        self.overlap = overlap
        self.crop = crop
        self.data_norm = data_norm
        self.dist_handling = dist_handling
        self.dist_scaling = dist_scaling
        self.seed_handling = seed_handling
        self.boundary_handling = boundary_handling
        self.instance_handling = instance_handling
        self.no_mask = no_mask
        self.no_img = no_img
        self.reduce_dim = reduce_dim
        
        # Read the filelist and construct full paths to each file
        self.image_groups = image_groups
        self.mask_groups = mask_groups
        self.image_noise_channel = image_noise_channel
        self.mask_noise_channel = mask_noise_channel
        self.noise_type = noise_type
        self.norm1 = 0
        self.norm2 = 1
        self.data_list = self._read_list()
        self.set_data_idx(0)
        
        # Get image statistics from up to 10 files
        if not self.no_img and 'image' in self.image_groups[0]:
            print('Getting statistics from images...')
            self.data_statistics = {'min':[], 'max':[], 'mean':[], 'std':[], 'perc01':[], 'perc99':[]}
            for file_pair in self.data_list[:10]:
                with h5py.File(file_pair[0], 'r') as f_handle:
                    image = f_handle[self.image_groups[0]][...].astype(np.float32)
                    self.data_statistics['min'].append(np.min(image))
                    self.data_statistics['max'].append(np.max(image))
                    self.data_statistics['mean'].append(np.mean(image))
                    self.data_statistics['std'].append(np.std(image))
                    perc01, perc99 = np.percentile(image, [1,99])
                    self.data_statistics['perc01'].append(perc01)
                    self.data_statistics['perc99'].append(perc99)
            
            # Construct data set statistics
            self.data_statistics['min'] = np.min(self.data_statistics['min'])
            self.data_statistics['max'] = np.max(self.data_statistics['max'])
            self.data_statistics['mean'] = np.mean(self.data_statistics['mean'])
            self.data_statistics['std'] = np.mean(self.data_statistics['std'])
            self.data_statistics['perc01'] = np.mean(self.data_statistics['perc01'])
            self.data_statistics['perc99'] = np.mean(self.data_statistics['perc99'])
            
            if self.data_norm == 'minmax':
                self.norm1 = self.data_statistics['min']
                self.norm2 = self.data_statistics['max']-self.data_statistics['min']
            elif self.data_norm == 'minmax_shifted':
                self.norm1 = 0.5*(self.data_statistics['max']+self.data_statistics['min'])
                self.norm2 = 0.5*(self.data_statistics['max']-self.data_statistics['min'])
            elif self.data_norm == 'meanstd':
                self.norm1 = self.data_statistics['mean']
                self.norm2 = self.data_statistics['std']
            elif self.data_norm == 'percentile':
                self.norm1 = self.data_statistics['perc01']
                self.norm2 = self.data_statistics['perc99']-self.data_statistics['perc01']
                    
        
    def _read_list(self):
        
        # Read the filelist and create full paths to each file
        filelist = []    
        with open(self.list_path, 'r') as f:
            reader = csv.reader(f, delimiter=';')
            for row in reader:
                if len(row)==0 or np.sum([len(r) for r in row])==0: continue
                row = [os.path.abspath(os.path.join(self.data_root, r)) for r in row]
                filelist.append(row)
                
        return filelist
    
    
    def get_fading_map(self):
               
        fading_map = np.ones(self.patch_size, dtype=np.float32)
        
        # Exclude crop region
        crop_masking = np.zeros_like(fading_map)
        crop_masking[self.crop[0]:self.patch_size[0]-self.crop[0],\
                     self.crop[1]:self.patch_size[1]-self.crop[1],\
                     self.crop[2]:self.patch_size[2]-self.crop[2]] = 1
        fading_map = fading_map * crop_masking
        
        # Pad with zero border to allow correct distance calculation
        pad_width = [(1,1) if d>1 else (0,0) for d in fading_map.shape]
        fading_map = np.pad(fading_map, pad_width)
            
        # Calculate distances
        fading_map = distance_transform_edt(fading_map).astype(np.float32)
        
        # Remove padded border
        fading_map = fading_map[pad_width[0][0]:fading_map.shape[0]-pad_width[0][1],\
                                pad_width[1][0]:fading_map.shape[1]-pad_width[1][1],\
                                pad_width[2][0]:fading_map.shape[2]-pad_width[2][1]]
        
        # Normalize
        fading_map = fading_map / fading_map.max()
        
        return fading_map
    
    
    def get_whole_image(self):
        
        with h5py.File(self.data_list[self.data_idx][0], 'r') as f_handle:
                image = f_handle[self.image_groups] [:]
        return image    
    
    
    def get_whole_mask(self, mask_groups=None): 
        
        if mask_groups is None:
            mask_groups = self.mask_groups
        if not isinstance(mask_groups, (list,tuple)):
            mask_groups = [mask_groups]
        
        mask = None
        with h5py.File(self.data_list[self.data_idx][1], 'r') as f_handle:
            for num_group, group_name in enumerate(mask_groups):
                mask_tmp = f_handle[group_name]
                if mask is None:
                    mask = np.zeros((len(mask_groups),)+mask_tmp.shape, dtype=np.float32)                
                mask[num_group,...] = mask_tmp
        return mask        
    
    
    def set_data_idx(self, idx):
        
        # Restrict the idx to the amount of data available
        idx = idx%len(self.data_list)
        self.data_idx = idx
        
        # Get the current data size
        if not self.no_img:
            with h5py.File(self.data_list[idx][0], 'r') as f_handle:
                image = f_handle[self.image_groups[0]]
                self.data_shape = image.shape[:3]
        elif not self.no_mask:
             with h5py.File(self.data_list[idx][1], 'r') as f_handle:
                mask = f_handle[self.mask_groups[0]]
                self.data_shape = mask.shape[:3]
        else:
            raise ValueError('Can not determine data shape!')
            
        # Calculate the position of each tile
        locations = []
        for i,p,o,c in zip(self.data_shape, self.patch_size, self.overlap, self.crop):
            # get starting coords
            coords = np.arange(np.ceil((i+o+c)/np.maximum(p-o-2*c,1)), dtype=np.int16)*np.maximum(p-o-2*c,1) -o-c
            locations.append(coords)
        self.locations = list(itertools.product(*locations))
        self.global_crop_before = np.abs(np.min(np.array(self.locations), axis=0))
        self.global_crop_after = np.array(self.data_shape) - np.max(np.array(self.locations), axis=0) - np.array(self.patch_size)
    
    
    def __len__(self):
        
        return len(self.locations)
    
    
    def _normalize(self, data, group_name):
        
        # Normalization
          
        if 'distance' in group_name:
            if self.dist_handling == 'float':
                data /= self.dist_scaling[0]
            elif self.dist_handling == 'bool':
                data = data<0
            elif self.dist_handling == 'bool_inv':
                data = data>=0
            elif self.dist_handling == 'exp':
                data = (data/self.dist_scaling[0])**3
            elif self.dist_handling == 'tanh':    
                foreground = np.float16(data>0)
                data = np.tanh(data/self.dist_scaling[0])*foreground + np.tanh(data/self.dist_scaling[1])*(1-foreground)
            
        elif 'seed' in group_name:                    
            if self.seed_handling == 'float':
                data = data.astype(np.float32)
                data = filters.gaussian_filter(data, 2)
                if np.max(data)>1e-4: data /= float(np.max(data))
            elif self.seed_handling == 'bool':
                data = data>0.1
            
        elif 'instance' in group_name or 'nuclei' in group_name:
            if self.instance_handling == 'bool':
                data = data>0
            
        elif 'boundary' in group_name:
            if self.boundary_handling == 'bool':
                data = data>0
                
        elif 'image' in group_name:
            data = data.astype(np.float32)
            data -= self.norm1
            data /= self.norm2
            if self.data_norm == 'minmax' or self.data_norm == 'percentile':
                data = np.clip(data, 1e-5, 1)
            elif self.data_norm == 'minmax_shifted':
                data = np.clip(data, -1, 1)
                
        return data
    
    
    def _get_noise(self, shape):
        
        if self.noise_type.lower() == 'gaussian':
            noise = np.random.randn(*shape)
        elif self.noise_type.lower() == 'rayleigh':
            noise = np.random.rayleigh(size=shape)
        elif self.noise_type.lower() == 'laplace':
            noise = np.random.laplace(size=shape)
        else:
            raise ValueError('Unknown noise distribution "{0}". Choose from (gaussian,rayleigh,laplace)'.format(self.noise_type))
        
        return noise.astype(np.float32)
    
    
    def __getitem__(self, idx):
        
        self.patch_start = np.array(self.locations[idx])
        self.patch_end = self.patch_start + np.array(self.patch_size) 
        
        pad_before = np.maximum(-self.patch_start, 0)
        pad_after = np.maximum(self.patch_end-np.array(self.data_shape), 0)
        pad_width = list(zip(pad_before, pad_after)) 
        
        slicing = tuple(map(slice, np.maximum(self.patch_start,0), self.patch_end))
        
        sample = {}
                
        # Load the mask patch
        if not self.no_mask:            
            mask = np.zeros((len(self.mask_groups),)+self.patch_size, dtype=np.float32)
            with h5py.File(self.data_list[self.data_idx][1], 'r') as f_handle:
                for num_group, group_name in enumerate(self.mask_groups):
                    mask_tmp = f_handle[group_name]
                    mask_tmp = mask_tmp[slicing]
                    
                    # Pad if neccessary
                    mask_tmp = np.pad(mask_tmp, pad_width, mode='reflect')
                    
                     # Normalization
                    mask_tmp = self._normalize(mask_tmp, group_name)
                    
                    # Add noise to current channel, if desired
                    if self.mask_noise_channel == num_group:
                        noise = self._get_noise(mask_tmp.shape)
                        mask_tmp += noise
                    
                    # Store current mask
                    mask[num_group,...] = mask_tmp
                    
            mask = mask.astype(np.float32)
            
            if self.reduce_dim:
                out_shape = [p for i,p in enumerate(mask.shape) if p!=1 or i==0]
                mask = np.reshape(mask, out_shape)
                
            if self.mask_noise_channel == -1:
                noise = self._get_noise((1,)+mask.shape[1:])
                mask = np.concatenate((mask, noise), axis=0)
            
            sample['mask'] = mask
            
            
        if not self.no_img:
            # Load the image patch
            image = np.zeros((len(self.image_groups),)+self.patch_size, dtype=np.float32)
            with h5py.File(self.data_list[self.data_idx][0], 'r') as f_handle:
                for num_group, group_name in enumerate(self.image_groups):
                    image_tmp = f_handle[group_name]   
                    image_tmp = image_tmp[slicing]
                    
                    # Pad if neccessary
                    image_tmp = np.pad(image_tmp, pad_width, mode='reflect')
                    
                    # Normalization
                    image_tmp = self._normalize(image_tmp, group_name)
                    
                    # Add noise to current channel, if desired
                    if self.image_noise_channel == num_group:
                        if self.mask_noise_channel==num_group and not self.no_mask:
                            pass # reuse the noise channel from the corresponding mask
                        else:
                            noise = self._get_noise(image_tmp.shape)
                        image_tmp += noise
                    
                    # Store current image
                    image[num_group,...] = image_tmp
            
            if self.reduce_dim:
                out_shape = [p for i,p in enumerate(image.shape) if p!=1 or i==0]
                image = np.reshape(image, out_shape)
                
            if self.image_noise_channel == -1:
                if self.mask_noise_channel==-1 and not self.no_mask:
                    pass # reuse the noise channel from the corresponding mask
                else:
                    noise = self._get_noise((1,)+image.shape[1:])
                image = np.concatenate((image, noise), axis=0)
            
            sample['image'] = image
        

                
        return sample
            
        