# Denoising Diffusion Probabilistic Models for Generation of Realistic Fully-Annotated Microscopy Image Data Sets
<br>
This repository contains code to simulated 2D/3D cellular structures and synthesize corresponding microscopy image data based on Denoising Diffusion Probabilistic Models (DDPM).
Sketches are generated to indicate cell shapes and structural characteristics, and they serve as a basis for the diffusion process to ultimately allow for the generation of fully-annotated microscopy image data sets without the need for human annotation effort.
Generated data sets are available at <a href=https://osf.io/dnp65/>OSF</a> and a preprint is available at <a href=https://arxiv.org/abs/2301.10227>arXiv/2301.10227</a>.
To access the trained models and get a showcase of the fully-simulated data sets, please visit to our <a href=https://transfer.lfb.rwth-aachen.de/CellDiffusion>website</a> (work in progress).<br><br>
<span style="white-space:nowrap"><img src="figures/example_data.png" alt="Diverse examplary synthetic data samples." align="middle" width="100%" /><br>
<span style="white-space:nowrap"><img src="figures/multi-channel.png" alt="Synthetic multi-channel data sample." align="middle" width="32.7%" /><img src="figures/overlapping_cells.png" alt="Synthetic data sample of overlapping cells." align="middle" width="32.7%" /><img src="figures/timeseries.gif" alt="Synthetic timeseries data sample." align="middle" width="32.7%" /></span></span><br>
<em>Exemplary synthetic samples from our experiments</em><br><br><br>


If you are using code or data, please cite the following work:
```
@article{eschweiler2022celldiffusion,
  title={Denoising Diffusion Probabilistic Models for Generation of Realistic Fully-Annotated Microscopy Image Data Sets},
  author={Dennis Eschweiler and Rüveyda Yilmaz and Matisse Baumann and Ina Laube and Rijo Roy and Abin Jose and Daniel Brückner and Johannes Stegmaier},
  journal={arXiv/2301.10227},
  year={2023}
}
```
<br><br><br>
We provide Jupyter Notebooks that give an overview of how to preprocess your data, train and apply the image generation process. The following gives a <em>very brief</em> overview of the general functionality, for more detailed examples we refer to the notebooks. 

## Data Preparation
The presented pipelines require the hdf5 file format for processing. Therefore, each image file has to be converted to hdf5, which can be done by using `utils.h5_converter.prepare_images`. Once all files have been contered, a list of those files has to be stored as a csv file to make them accessible by the processing pipelines. This can be done by using `utils.csv_generator.create_csv`. A more detailed explanation is given in `jupyter_preparation_script.ipynb`.


## Diffusion Model Training and Application
To use the proposed pipeline to either train or apply your models, make sure to adapt all parameters in the pipeline files `models/DiffusionModel3D` or `models/DiffusionModel2D`, and in the training script `train_script.py` or application script `apply_script_diffusion.py`. Alternatively, it is recommended to use command line arguments to adapt all parameters at once. A more detailed explanation is given in `jupyter_train_script.ipynb` and `jupyter_apply_script.ipynb`..


## Simulation of Cellular Structures and Sketch Generation
Since the proposed approach is working in a very intuitive manner, sketches can generally be created in any arbitrary way. We mainly focused on using and adapting simulation techniques proposed in <a href='https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0260509'>3D fluorescence microscopy data synthesis for segmentation and benchmarking</a>. Nevertheless, the functionality used in this work can be found in `utils.synthetic_cell_membrane_masks` and `utils.synthetic_cell_nuclei_masks` for cellular membranes and nuclei, respectively.
