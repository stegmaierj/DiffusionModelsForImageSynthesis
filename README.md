# Denoising Diffusion Probabilistic Models for Generation of Realistic Fully-Annotated Microscopy Image Data Sets
<br>
This repository contains code to simulated 2D/3D cellular structures and synthesize corresponding microscopy image data based on Denoising Diffusion Probabilistic Models (DDPM).
Sketches are generated to indicate cell shapes and structural characteristics, and they serve as a basis for the diffusion process to ultimately allow for the generation of fully-annotated microscopy image data sets without the need for human annotation effort.
Generated data sets are available at <a href=https://osf.io/dnp65/>OSF</a> and a preprint is available at <a href=https://arxiv.org/abs/2301.10227>arXiv/2301.10227</a>.<br><br>
<img src="figures/example_data.png" alt="Examplary sketches and corresponding synthetic data." align="middle" /><em>Exemplary synthetic samples from our experiments</em><br><br><br>


If you are using code or data, please cite the following work:
```
@article{eschweiler2022celldiffusion,
  title={Denoising Diffusion Probabilistic Models for Generation of Realistic Fully-Annotated Microscopy Image Data Sets},
  author={Dennis Eschweiler and Johannes Stegmaier},
  journal={arXiv/2301.10227},
  year={2023}
}
```
<br><br><br>
The following gives a <em>very brief</em> overview of the functionality. 
More details and more thorough instructions will follow shortly.

## Diffusion Model Training
To prepare your own data for training of the diffusion model, please use `utils.h5_conveter.prepapre_images` to convert the image data to the hdf5 format and use `utils.csv_generator.create_csv` to create a csv file listing all training data.
Make sure to adapt all parameters in the pipeline files `models/DiffusionModel3D` and `models/DiffusionModel2D`, and in the training script `train_script.py`, or use command line arguments to adapt all parameters at once.
Run `train_script.py` to train the model.


## Diffusion Model Application
To prepare you own data for application of the diffusion model, please use `utils.h5_conveter.prepapre_masks` to convert your data to the hdf5 format and apply `prepare_diffusionmasks.py` to generate the corresponding sketches.
Create a csv file listing all testing data using `utils.csv_generator.create_csv`.
Again make sure to adapt all parameters in the pipeline files `models/DiffusionModel3D` and `models/DiffusionModel2D`, and in the training script `apply_script.py`, or use command line arguments to adapt all parameters at once.
Run `apply_script_diffusion.py` to train the model.


## Simulation of Cellular Structures
To simulate cellular structures use `utils.synthetic_cell_membrane_masks.generate_data` and `utils.synthetic_cell_nuclei_masks.generate_data` for cellular membranes and nuclei, respectively.
Sketches are generated by applying `prepare_diffusionmasks.py`.
