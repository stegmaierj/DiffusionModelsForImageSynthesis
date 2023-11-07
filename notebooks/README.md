By providing jupyter notebooks we try to give a clear overview of how the data preparation, training and application pipelines work.
Parameters that are important to these pipelines and might require further clarification are listed and explained in the following.

#### General Network Parameters
- `in_channels`: (int) Number of input channels, including an additional noise channel
- `out_channels`: (int) Number of output channels
- `feat_channels`: (int) Number of feature channels in the first block of the UNet backbone. Other channel numbers are derived from this
- `t_channels`: (int) Number of channels used for timestep encoding
- `patch_size`: (int+) Size of the patches. For 2D make sure to still provide a 3D patch size with a leading 1 with (1,y,x)
- `data_root`: (str) Directory of the image data
- `train_list`, `test_list`, `val_list`: (str) Path to the csv file listing all image data that should be used for training/testing/validation. The concatenation of the _data_root_ parameter and the entries of this list should give the full path to each file.
- `image_groups`: (str+) List of group names listed in the hdf5 file that should be used
- `num_timesteps`: (int) Number of total diffusion timesteps
- `diffusion_schedule`: (str) Noise schedule used during the diffusion forward process

#### Training-specific Parameters
- `output_path`: (str) Directory for saving the model
- `log_path`: (str) Directory for saving the log files
- `no_resume`: (bool) Flag to not resume training from an existing checkpoint with the same name as the current training experiment
- `pretrained`: (str) Explicit definition of a pretrained model for further training
- `epochs`: (int) Number of training epochs
- `samples_per_epoch`: (int) Number of samples used in one training epoch. Set to -1 to use all available samples

#### Application-specific Parameters
- `output_path`: (str) Directory for saving the generated image data
- `ckpt_path`: (str) Path to the trained model checkpoint file
- `overlap`: (int+) Tuple of overlap of neighbouring patches during the patch-based application, defined in z,y,x
- `crop`: (int+) Tuple of cropped patch borders to avoid potential border artifacts, defined in z,y,x
- `clip`: (int+) Intensity range the output gets clipped to
- `timesteps_start`: (int) Timestep for initiating the diffusion backward process
- `timesteps_save`: (int) Interval of timesteps after which (intermediate) results are saved
- `timesteps_step`: (int) Number of timesteps skipped between consecutive iterations of the diffusion backward process
- `blur_sigma`: (int) Sigma of the Gaussian blurring applied before starting the diffusion forward process
- `num_files`: (int) Number of files that should be processed. Set to -1 to process all available files
