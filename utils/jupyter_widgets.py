#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import ipywidgets as wg
from IPython.display import display 
style={'description_width': 'initial'}



def get_pipelin_widget():
    
    pipeline = wg.Dropdown(
        options=['DiffusionModel3d', 'DiffusionModel2d'],
        description='Select Pipeline:', style=style)
    
    return pipeline


def get_execution_widgets():
    
    wg_execute = wg.Checkbox(description='Execute Now!', value=True, style=style)
    wg_arguments = wg.Checkbox(description='Get Command Line Arguments', value=True, style=style)
    
    return [wg_execute, wg_arguments]
    

def get_synthesizer_widget():
    
    pipeline = wg.Dropdown(
        options=['SyntheticTRIC', 'SyntheticCE', 'Synthetic2DGOWT1', 'Synthetic2DHeLa', 'SyntheticMeristem'],
        description='Select Synthesizer:', style=style)
    
    return pipeline



def get_parameter_widgets(param_dict):
    
    param_names = []
    widget_list = []
    test_related = []
    
    for key in param_dict.keys():
        ### Script Parameter
        
        if key == 'output_path':
            widget_list.append(wg.Text(description='Output Path:', value=param_dict[key], style=style))
            param_names.append('--'+key)
            test_related.append(True)
        if key == 'log_path':
            widget_list.append(wg.Text(description='Log Path:', value=param_dict[key], style=style))
            param_names.append('--'+key)
            test_related.append(False)
        if key == 'gpus':
            widget_list.append(wg.IntSlider(description = 'Use GPU:', min=0, max=1, value=param_dict[key], style=style))
            param_names.append('--'+key)
            test_related.append(True)
        if key == 'no_resume':
            widget_list.append(wg.Checkbox(description='Resume:', value=not param_dict[key], style=style))
            param_names.append('--'+key)
            test_related.append(False)
        if key == 'pretrained':
            widget_list.append(wg.Text(description='Path To The Pretrained Model:', value=param_dict[key], style=style))
            param_names.append('--'+key)
            test_related.append(False)
        if key == 'augmentations':
            widget_list.append(wg.Text(description='Augmentation Dictionary File:', value=param_dict[key], style=style))
            param_names.append('--'+key)
            test_related.append(False)
        if key == 'epochs':
            widget_list.append(wg.BoundedIntText(description='Epochs:', min=1, max=10000, value=param_dict[key], style=style))
            param_names.append('--'+key)
            test_related.append(False)
           
           
        ### Network Parameter
        
        if key == 'backbone':
            widget_list.append(wg.Dropdown(description='Network Architecture:', options=['UNet3D_PixelShuffle_inject', 'UNet2D_PixelShuffle_inject'], value=param_dict[key], layout={'width': 'max-content'}, style=style))
            param_names.append('--'+key)
            test_related.append(True)
        if key == 'in_channels':
            widget_list.append(wg.BoundedIntText(description='Input Channels:', value=param_dict[key], min=1, max=10000, style=style))
            param_names.append('--'+key)
            test_related.append(True)
        if key == 'out_channels':
            widget_list.append(wg.BoundedIntText(description='Output Channels:', value=param_dict[key], min=1, max=10000, style=style))
            param_names.append('--'+key)
            test_related.append(True)
        if key == 'feat_channels':
            widget_list.append(wg.BoundedIntText(description='Feature Channels:', value=param_dict[key], min=2, max=10000, style=style))
            param_names.append('--'+key)
            test_related.append(True)
        if key == 'patch_size':
            if not param_dict[key] is str:
                param_dict[key] = ' '.join([str(p) for p in param_dict[key]])
            widget_list.append(wg.Text(description='Patch Size (z,y,x):', value=param_dict[key], style=style))
            param_names.append('--'+key)
            test_related.append(True)
        if key == 'out_activation':
            widget_list.append(wg.Dropdown(description='Output Activation:', options=['tanh', 'sigmoid', 'hardtanh', 'relu', 'leakyrelu', 'none'], value=param_dict[key], layout={'width': 'max-content'}, style=style))
            param_names.append('--'+key)
            test_related.append(True)
        if key == 'layer_norm':
            widget_list.append(wg.Dropdown(description='Layer Normalization:', options=['instance', 'batch', 'none'], value=param_dict[key], layout={'width': 'max-content'}, style=style))
            param_names.append('--'+key)
            test_related.append(True)
        if key == 't_channels':
            widget_list.append(wg.BoundedIntText(description='T Channels:', value=param_dict[key], min=1, max=10000, style=style))
            param_names.append('--'+key)
            test_related.append(True)
            
        ### Data Parameter
            
        if key == 'data_norm':
            widget_list.append(wg.Dropdown(description='Data Normalization:', options=['percentile', 'minmax', 'meanstd', 'minmax_shifted', 'none'], value=param_dict[key], layout={'width': 'max-content'}, style=style))
            param_names.append('--'+key)
            test_related.append(True)
        if key == 'data_root':
            widget_list.append(wg.Text(value=param_dict[key], description='Data Root:', style=style))
            param_names.append('--'+key)
            test_related.append(True)
        if key == 'train_list':
            widget_list.append(wg.Text(value=param_dict[key], description='Train List:', style=style))
            param_names.append('--'+key)
            test_related.append(True)
        if key == 'test_list':
            widget_list.append(wg.Text(value=param_dict[key], description='Test List:', style=style))
            param_names.append('--'+key)
            test_related.append(True)
        if key == 'val_list':
            widget_list.append(wg.Text(value=param_dict[key], description='Validation List:', style=style))
            param_names.append('--'+key)
            test_related.append(True)
        if key == 'image_groups':
            if not param_dict[key] is str:
                param_dict[key] = ' '.join([str(p) for p in param_dict[key]])
            widget_list.append(wg.Text(description='Image Groups:', value=param_dict[key], style=style))
            param_names.append('--'+key)
            test_related.append(True)
        if key == 'mask_groups':
            if not param_dict[key] is str:
                param_dict[key] = ' '.join([str(p) for p in param_dict[key]])
            widget_list.append(wg.Text(description='Mask Groups:', value=param_dict[key], style=style))
            param_names.append('--'+key)
            test_related.append(True)
        if key == 'dist_handling':
            widget_list.append(wg.Dropdown(description='Distance Handling:', options=['float', 'bool', 'bool_inv', 'exp', 'tanh', 'none'], value=param_dict[key], layout={'width': 'max-content'}, style=style))
            param_names.append('--'+key)
            test_related.append(True)
        if key == 'dist_scaling':
            if not param_dict[key] is str:
                param_dict[key] = ' '.join([str(p) for p in param_dict[key]])
            widget_list.append(wg.Dropdown(description='Distance Scaling:', value=param_dict[key], style=style))
            param_names.append('--'+key)
            test_related.append(True)
        if key == 'seed_handling':
            widget_list.append(wg.Dropdown(description='Seed Handling:', options=['float', 'bool', 'none'], value=param_dict[key], layout={'width': 'max-content'}, style=style))
            param_names.append('--'+key)
            test_related.append(True)
        if key == 'boundary_handling':
            widget_list.append(wg.Dropdown(description='Boundary Handling:', options=['bool', 'none'], value=param_dict[key], layout={'width': 'max-content'}, style=style))
            param_names.append('--'+key)
            test_related.append(True)
        if key == 'instance_handling':
            widget_list.append(wg.Dropdown(description='Instance Handling:', options=['bool', 'none'], value=param_dict[key], layout={'width': 'max-content'}, style=style))
            param_names.append('--'+key)
            test_related.append(True)
        if key == 'strides':
            if not param_dict[key] is str:
                param_dict[key] = ' '.join([str(p) for p in param_dict[key]])
            widget_list.append(wg.Dropdown(description='Strides:', value=param_dict[key], style=style))
            param_names.append('--'+key)
            test_related.append(True)
        if key == 'sh_order':
            widget_list.append(wg.BoundedIntText(description='SH Order:', value=param_dict[key], min=0, max=10000, style=style))
            param_names.append('--'+key)
            test_related.append(True)
        if key == 'mean_low':
            widget_list.append(wg.BoundedFloatText(description='Sampling Mean Weight Low:', value=param_dict[key], min=0, max=1000000, style=style))
            param_names.append('--'+key)
            test_related.append(True)
        if key == 'mean_high':
            widget_list.append(wg.BoundedFloatText(description='Sampling Mean Weight High:', value=param_dict[key], min=0, max=1000000, style=style))
            param_names.append('--'+key)
            test_related.append(True)
        if key == 'var_high':
            widget_list.append(wg.BoundedFloatText(description='Sampling Variance Weight High:', value=param_dict[key], min=0, max=1000000, style=style))
            param_names.append('--'+key)
            test_related.append(True)
        if key == 'variance_levels':
            widget_list.append(wg.BoundedIntText(description='Sampling Variance Levels:', value=param_dict[key], min=0, max=10000, style=style))
            param_names.append('--'+key)
            test_related.append(True)
        if key == 'strategy':
            widget_list.append(wg.Dropdown(description='Sampling Variance Strategy:', options=['random', 'structured'], value=param_dict[key], layout={'width': 'max-content'}, style=style))
            param_names.append('--'+key)
            test_related.append(True)
        if key == 'image_noise_channel':
            widget_list.append(wg.BoundedIntText(description='Image Noise Channel:', value=param_dict[key], min=-5, max=5, style=style))
            param_names.append('--'+key)
            test_related.append(True)
        if key == 'mask_noise_channel':
            widget_list.append(wg.BoundedIntText(description='Mask Noise Channel:', value=param_dict[key], min=-5, max=5, style=style))
            param_names.append('--'+key)
            test_related.append(True)
        if key == 'noise_type':
            widget_list.append(wg.Dropdown(description='Noise Type:', options=['gaussian', 'rayleigh', 'laplace'], value=param_dict[key], layout={'width': 'max-content'}, style=style))
            param_names.append('--'+key)
            test_related.append(True)
        
        ### Training Parameter
        
        if key == 'samples_per_epoch':
            widget_list.append(wg.BoundedIntText(description='Samples Per Epoch:', value=param_dict[key], min=-1, max=1000000, style=style))
            param_names.append('--'+key)
            test_related.append(True)
        if key == 'batch_size':
            widget_list.append(wg.BoundedIntText(description='Batch Size:', value=param_dict[key], min=1, max=1000000, style=style))
            param_names.append('--'+key)
            test_related.append(True)
        if key == 'learning_rate':
            widget_list.append(wg.BoundedFloatText(description='Learning Rate:', value=param_dict[key], min=0, max=1000000, style=style))
            param_names.append('--'+key)
            test_related.append(True)
        if key == 'background_weight':
            widget_list.append(wg.BoundedFloatText(description='Background Weight:', value=param_dict[key], min=0, max=1000000, style=style))
            param_names.append('--'+key)
            test_related.append(True)
        if key == 'seed_weight':
            widget_list.append(wg.BoundedFloatText(description='Seed Weight:', value=param_dict[key], min=0, max=1000000, style=style))
            param_names.append('--'+key)
            test_related.append(True)
        if key == 'boundary_weight':
            widget_list.append(wg.BoundedFloatText(description='Boundary Weight:', value=param_dict[key], min=0, max=1000000, style=style))
            param_names.append('--'+key)
            test_related.append(True)
        if key == 'flow_weight':
            widget_list.append(wg.BoundedFloatText(description='Flow Weight:', value=param_dict[key], min=0, max=1000000, style=style))
            param_names.append('--'+key)
            test_related.append(True)
        if key == 'centroid_weight':
            widget_list.append(wg.BoundedFloatText(description='Centroid Weight:', value=param_dict[key], min=0, max=1000000, style=style))
            param_names.append('--'+key)
            test_related.append(True)
        if key == 'encoding_weight':
            widget_list.append(wg.BoundedFloatText(description='Encoding Weight:', value=param_dict[key], min=0, max=1000000, style=style))
            param_names.append('--'+key)
            test_related.append(True)
        if key == 'robustness_weight':
            widget_list.append(wg.BoundedFloatText(description='Robustness Weight:', value=param_dict[key], min=0, max=1000000, style=style))
            param_names.append('--'+key)
            test_related.append(True)
        if key == 'variance_interval':
            widget_list.append(wg.BoundedIntText(description='Sampling Variance Interval:', value=param_dict[key], min=0, max=10000, style=style))
            param_names.append('--'+key)
            test_related.append(True)
        if key == 'ada_update_period':
            widget_list.append(wg.BoundedIntText(description='ADA Update Period:', value=param_dict[key], min=0, max=10000, style=style))
            param_names.append('--'+key)
            test_related.append(True)
        if key == 'ada_update':
            widget_list.append(wg.BoundedFloatText(description='ADA Update Step:', value=param_dict[key], min=0, max=1000000, style=style))
            param_names.append('--'+key)
            test_related.append(True)
        if key == 'ada_target':
            widget_list.append(wg.BoundedFloatText(description='ADA Target:', value=param_dict[key], min=0, max=1000000, style=style))
            param_names.append('--'+key)
            test_related.append(True)
        if key == 'num_samples':
            widget_list.append(wg.BoundedIntText(description='Number Of Samples:', value=param_dict[key], min=0, max=10000, style=style))
            param_names.append('--'+key)
            test_related.append(True)
        
        # diffusion parameter
        if key == 'num_timesteps':
            widget_list.append(wg.BoundedIntText(description='Number Of Timesteps:', value=param_dict[key], min=0, max=1000, style=style))
            param_names.append('--'+key)
            test_related.append(True)
        if key == 'diffusion_schedule':
            widget_list.append(wg.Dropdown(description='Diffusion Schedule:', options=['cosine', 'linear', 'quadratic', 'sigmoid'], value=param_dict[key], layout={'width': 'max-content'}, style=style))
            param_names.append('--'+key)
            test_related.append(True)
    return param_names, widget_list, test_related



def get_apply_parameter_widgets(param_dict):
    
    param_names = []
    widget_list = []

    for key in param_dict.keys():
    
    
        ### Script Parameter
        
        if key == 'output_path':
            widget_list.append(wg.Text(description='Output Path:', value=param_dict[key], style=style))
            param_names.append('--'+key)
        elif key == 'ckpt_path':
            widget_list.append(wg.Text(description='Checkpoint Path:', value=param_dict[key], style=style))
            param_names.append('--'+key)
        elif key == 'gpus':
            widget_list.append(wg.IntSlider(description = 'Use GPU:', min=0, max=1, value=param_dict[key], style=style))
            param_names.append('--'+key)
        elif key == 'distributed_backend':
            widget_list.append(wg.Dropdown(description='Distributed Backend:', options=['dp', 'ddp', 'ddp2'], value=param_dict[key], layout={'width': 'max-content'}, style=style))
            param_names.append('--'+key)
        elif key == 'overlap':
            if not param_dict[key] is str:
                param_dict[key] = ' '.join([str(p) for p in param_dict[key]])
            widget_list.append(wg.Text(description='Overlap (z,y,x):', value=param_dict[key], style=style))
            param_names.append('--'+key)
        elif key == 'crop':
            if not param_dict[key] is str:
                param_dict[key] = ' '.join([str(p) for p in param_dict[key]])
            widget_list.append(wg.Text(description='Crop (z,y,x):', value=param_dict[key], style=style))
            param_names.append('--'+key)
        elif key == 'input_batch':
            widget_list.append(wg.Dropdown(description='Input Batch:', options=['image', 'mask'], value=param_dict[key], layout={'width': 'max-content'}, style=style))
            param_names.append('--'+key)
        elif key == 'clip':
            if not param_dict[key] is str:
                param_dict[key] = ' '.join([str(p) for p in param_dict[key]])
            widget_list.append(wg.Text(description='Clip (min, max):', value=param_dict[key], style=style))
            param_names.append('--'+key)
        elif key == 'num_files':
            widget_list.append(wg.BoundedIntText(description='Number of Files:', value=param_dict[key], min=-1, max=10000, style=style))
            param_names.append('--'+key)
        elif key == 'add_noise_channel':
            widget_list.append(wg.BoundedIntText(description='Noise Channel:', value=param_dict[key], min=-2, max=10000, style=style))
            param_names.append('--'+key)
        elif key == 'theta_phi_sampling':
            widget_list.append(wg.Text(description='Angular Sampling File Path:', value=param_dict[key], style=style))
            param_names.append('--'+key)

        ### Network Parameter
        elif key == 'out_channels':
            widget_list.append(wg.BoundedIntText(description='Output Channels:', value=param_dict[key], min=1, max=10000, style=style))
            param_names.append('--'+key)
        elif key == 'patch_size':
            if not param_dict[key] is str:
                param_dict[key] = ' '.join([str(p) for p in param_dict[key]])
            widget_list.append(wg.Text(description='Patch Size (z,y,x):', value=param_dict[key], style=style))
            param_names.append('--'+key)
        elif key == 'resolution_weights':
            if not param_dict[key] is str:
                param_dict[key] = ' '.join([str(p) for p in param_dict[key]])
            widget_list.append(wg.Text(description='Prediction Weights at Each Resolution:', value=param_dict[key], style=style))
            param_names.append('--'+key)
        elif key == 'centroid_thresh':
            widget_list.append(wg.BoundedFloatText(description='Centroid Threshold:', value=param_dict[key], min=0, max=100, style=style))
            param_names.append('--'+key)
        elif key == 'minsize':
            widget_list.append(wg.BoundedIntText(description='The Minimum Cell Size:', value=param_dict[key], min=0, max=100, style=style))
            param_names.append('--'+key)
        elif key == 'maxsize':
            widget_list.append(wg.BoundedIntText(description='The Maximum Cell Size:', value=param_dict[key], min=0, max=100, style=style))
            param_names.append('--'+key)
        elif key == 'use_watershed':
            widget_list.append(wg.Checkbox(description='Use Watershed:', value=param_dict[key], style=style))
            param_names.append('--'+key)
        elif key == 'use_sizefilter':
            widget_list.append(wg.Checkbox(description='Use Size Filter:', value=param_dict[key], style=style))
            param_names.append('--'+key)
        elif key == 'use_nms':
            widget_list.append(wg.Checkbox(description='Use non-Maximum Suppression:', value=param_dict[key], style=style))
            param_names.append('--'+key)
        ### Data Parameter
        elif key == 'data_root':
            widget_list.append(wg.Text(value=param_dict[key], description='Data Root:', style=style))
            param_names.append('--'+key)
        elif key == 'test_list':
            widget_list.append(wg.Text(value=param_dict[key], description='Test List:', style=style))
            param_names.append('--'+key)
        elif key == 'image_groups':
            if not param_dict[key] is str:
                param_dict[key] = ' '.join([str(p) for p in param_dict[key]])
            widget_list.append(wg.Text(description='Image Groups:', value=param_dict[key], style=style))
            param_names.append('--'+key)
        elif key == 'mask_groups':
            if not param_dict[key] is str:
                param_dict[key] = ' '.join([str(p) for p in param_dict[key]])
            widget_list.append(wg.Text(description='Mask Groups:', value=param_dict[key], style=style))
            param_names.append('--'+key)
        elif key == 'sh_order':
            widget_list.append(wg.BoundedIntText(description='SH Order:', value=param_dict[key], min=0, max=10000, style=style))
            param_names.append('--'+key)
        
        # diffusion parameter
        elif key == 'timesteps_start':
            widget_list.append(wg.BoundedIntText(description='Timestep to Start the Reverse Process:', value=param_dict[key], min=0, max=1000, style=style))
            param_names.append('--'+key)
        elif key == 'timesteps_save':
            widget_list.append(wg.BoundedIntText(description='Number of Timesteps Between Saves:', value=param_dict[key], min=0, max=1000, style=style))
            param_names.append('--'+key)
        elif key == 'timesteps_step':
            widget_list.append(wg.BoundedIntText(description='Timesteps Skipped Between Iterations:', value=param_dict[key], min=0, max=1000, style=style))
            param_names.append('--'+key)
        elif key == 'blur_sigma':
            widget_list.append(wg.BoundedIntText(description='Sigma for Gaussian Blurring of Inputs:', value=param_dict[key], min=0, max=5, style=style))
            param_names.append('--'+key)
        elif key == 'num_timesteps':
            widget_list.append(wg.BoundedIntText(description='Total Number Of Training Timesteps:', value=param_dict[key], min=0, max=1000, style=style))
            param_names.append('--'+key)
        elif key == 'diffusion_schedule':
            widget_list.append(wg.Dropdown(description='Diffusion Schedule:', options=['cosine', 'linear', 'quadratic', 'sigmoid'], value=param_dict[key], layout={'width': 'max-content'}, style=style))
            param_names.append('--'+key)

    return param_names, widget_list


def get_sim_parameter_widgets(param_dict):
    
    param_names = []
    widget_list = []
    
    for key in param_dict.keys():
        if key == 'save_path':
            widget_list.append(wg.Text(description='Save Path:', value=param_dict[key], style=style))
            param_names.append('--'+key)
        elif key == 'experiment_name':
            widget_list.append(wg.Text(description='Experiment Name:', value=param_dict[key], style=style))
            param_names.append('--'+key)
        elif key == 'num_imgs' or key == 'img_count':
            widget_list.append(wg.BoundedIntText(description='Number of Images:', value=param_dict[key], min=1, max=1000, style=style))
            param_names.append('--'+key)
        
        # Nuclei masks
        elif key == 'img_shape':
            if not param_dict[key] is str:
                param_dict[key] = ' '.join([str(p) for p in param_dict[key]])
            widget_list.append(wg.Text(description='Image Shape:', value=param_dict[key], style=style))
            param_names.append('--'+key)
        elif key == 'max_radius':
            widget_list.append(wg.BoundedIntText(description='Maximum Radius:', value=param_dict[key], min=1, max=1000, style=style))
            param_names.append('--'+key)
        elif key == 'min_radius':
            widget_list.append(wg.BoundedIntText(description='Minimum Radius:', value=param_dict[key], min=1, max=100, style=style))
            param_names.append('--'+key)
        elif key == 'radius_range':
            widget_list.append(wg.BoundedIntText(description='Radius Range:', value=param_dict[key], min=-100, max=100, style=style))
            param_names.append('--'+key)
        elif key == 'sh_order':
            widget_list.append(wg.BoundedIntText(description='Sh Order :', value=param_dict[key], min=1, max=100, style=style))
            param_names.append('--'+key)
        elif key == 'smooth_std':
            widget_list.append(wg.BoundedFloatText(description='Smoothing Std :', value=param_dict[key], min=0, max=100, style=style))
            param_names.append('--'+key)
        elif key == 'noise_std':
            widget_list.append(wg.BoundedFloatText(description='Noise Std :', value=param_dict[key], min=0, max=100, style=style))
            param_names.append('--'+key)
        elif key == 'noise_mean':
            widget_list.append(wg.BoundedFloatText(description='Noise Mean :', value=param_dict[key], min=0, max=100, style=style))
            param_names.append('--'+key)
        elif key == 'position_std':
            widget_list.append(wg.BoundedFloatText(description='Position Std :', value=param_dict[key], min=0, max=100, style=style))
            param_names.append('--'+key)
        elif key == 'num_cells':
            widget_list.append(wg.BoundedIntText(description='Number of Cells :', value=param_dict[key], min=1, max=1000, style=style))
            param_names.append('--'+key)
        elif key == 'num_cells_range':
            widget_list.append(wg.BoundedIntText(description='Number of Cells Range:', value=param_dict[key], min=1, max=1000, style=style))
            param_names.append('--'+key)
        elif key == 'circularity':
            widget_list.append(wg.BoundedFloatText(description='Circularity :', value=param_dict[key], min=0, max=100, style=style))
            param_names.append('--'+key)
        elif key == 'generate_images':
            widget_list.append(wg.Checkbox(description='Generate Images ', value=False, style=style))
            param_names.append('--'+key)
        elif key == 'theta_phi_sampling_file':
            widget_list.append(wg.Text(value=param_dict[key], description='Theta Phi Sampling File:', style=style))
            param_names.append('--'+key)
        elif key == 'cell_elongation':
            widget_list.append(wg.BoundedFloatText(description='Cell Elongation :', value=param_dict[key], min=0, max=100, style=style))
            param_names.append('--'+key)
        elif key == 'z_anisotropy':
            widget_list.append(wg.BoundedFloatText(description='Z Anisotropy :', value=param_dict[key], min=0, max=100, style=style))
            param_names.append('--'+key)
        elif key == 'irregularity_extend':
            widget_list.append(wg.BoundedFloatText(description='Irregularity Extend :', value=param_dict[key], min=0, max=100, style=style))
            param_names.append('--'+key)

        # Membrane masks
        elif key == 'gridsize':
            if not param_dict[key] is str:
                param_dict[key] = ' '.join([str(p) for p in param_dict[key]])
            widget_list.append(wg.Text(description='Grid Size:', value=param_dict[key], style=style))
            param_names.append('--'+key)
        elif key == 'distance_weight':
            widget_list.append(wg.BoundedFloatText(description='Distance Weight :', value=param_dict[key], min=0, max=10, style=style))
            param_names.append('--'+key)
        elif key == 'morph_radius':
            widget_list.append(wg.BoundedIntText(description='Morph Radius:', value=param_dict[key], min=1, max=1000, style=style))
            param_names.append('--'+key)
        elif key == 'weights':
            widget_list.append(wg.Text(description='Cell Weights:', value=param_dict[key], style=style))
            param_names.append('--'+key)
        elif key == 'cell_density':
            widget_list.append(wg.BoundedFloatText(description='Cell Density:', value=param_dict[key], min=0, max=10, style=style))
            param_names.append('--'+key)
        elif key == 'cell_density_decay':
            widget_list.append(wg.BoundedFloatText(description='Cell Density Decay:', value=param_dict[key], min=0, max=10, style=style))
            param_names.append('--'+key)
        elif key == 'cell_position_smoothness':
            widget_list.append(wg.BoundedIntText(description='Cell Position Smoothness:', value=param_dict[key], min=1, max=1000, style=style))
            param_names.append('--'+key)
        elif key == 'ring_density':
            widget_list.append(wg.BoundedFloatText(description='Ring Density:', value=param_dict[key], min=0, max=10, style=style))
            param_names.append('--'+key)
        elif key == 'ring_density_decay':
            widget_list.append(wg.BoundedFloatText(description='Ring Density Decay:', value=param_dict[key], min=0, max=10, style=style))
            param_names.append('--'+key)
        elif key == 'angular_sampling_file':
            widget_list.append(wg.Text(description='Angular Sampling File:', value=param_dict[key], style=style))
            param_names.append('--'+key)
        elif key == 'specimen_sampling_file':
            widget_list.append(wg.Text(description='Specimen Sampling File:', value=param_dict[key], style=style))
            param_names.append('--'+key)
    return param_names, widget_list

