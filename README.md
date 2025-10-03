[![Docker](https://github.com/mfmceneaney/lambdaml/actions/workflows/docker-image.yml/badge.svg)](https://github.com/mfmceneaney/lambdaml/actions/workflows/docker-image.yml)

# $\Lambda$ ML

This is a project to train Graph Neural Networks (GNNs) for $\Lambda$ hyperon event identification at [CLAS12](https://www.jlab.org/physics/hall-b/clas12).
[Domain Adversarial](https://arxiv.org/abs/1505.07818), [Contrastive Adaptation](http://arxiv.org/abs/1901.00976), and [TIToK](https://www.sciencedirect.com/science/article/pii/S0893608023002137) approaches are implemented for adapting models to target data domains.

## :green_circle: Installation

You can use either a container image via [Docker](https://www.docker.com) or [Singularity/Apptainer](https://github.com/apptainer/apptainer) or install manually assuming you have python installed.

### Installation Via Docker

Begin by cloning the repository:
```bash
git clone https://github.com/mfmceneaney/lambdaml.git
```

Then, build the project (this may take a while).
```bash
docker build -f /path/to/lambdaml/Dockerfile.cpu -t lambdaml-project /path/to/lambdaml #Note: There is also a cuda Dockerfile.
```
After successfully building, run the project with:
```bash
docker run --rm -it lambdaml-project
```
The `--rm` option tells docker to remove the container and its data once it is shut down.
To retain the container data though, you can mount a local directory (`src`) to a directory (`dst`)
inside the container with the following:
```bash
docker run --rm -it -v <src>:<dst> lambdaml-project
```
To link all available GPUs on your node, e.g., in a SLURM job use the `--gpus all` option.
```bash
docker run --rm -it --gpus all lambdaml-project
```
If you really only need to run a single python script in the container and then exit, for example, for a SLURM job, you can do that too.
```bash
docker run --rm lambdaml-project python3 </path/to/my/python/sript.py>
```
Once you start the container you should have the following environment variables:
- `LAMBDAML_HOME`
- `LAMBDAML_REGISTRY`

If you have input data directories and output data directories for your preprocessing or training pipelines, you can mount several directories.
```bash
docker run --rm -it -v /path/to/lambdaml:/usr/src/lambdaml -v /path/for/input/files:/data -v /path/for/out/files:/out lambdaml-project-cu129
```
For use with CUDA, see the bit about installing PyTorch-Geometric on an HPC cluster below as well.

<details>
<summary>:red_circle: Running in a SLURM Job</summary>

It is very hard to access the different volumes of a HPC cluster from Docker, so use singularity instead.
Download the PyTorch-Geometric packages and copy them to `/path/to/lambdaml/pyg_packages`. Then, build the container with
```bash
singularity build lambdaml-cu129.sif Singularity.def.cu129
```
Then run the container, binding to some volumes on your cluster, with
```bash
singularity exec -B /volatile,/path/to/lambdaml:/usr/src/lambdaml lambdaml-cu129.sif bash
```
Or, if you just need to run a python script within the container
```bash
singularity exec -B /volatile,/path/to/lambdaml:/usr/src/lambdaml lambdaml-cu129.sif python3 /usr/src/lambdaml/pyscripts/<SCRIPT>.py --help
```
Also, when running t-SNE latent space visualization on HPC, you may get the following error due to the fact that you can have many more cores available than the precompiled allowed numbers for OpenBLAS libraries used in numpy and torch.
```bash
OpenBLAS warning: precompiled NUM_THREADS exceeded, adding auxiliary array for thread metadata.
To avoid this warning, please rebuild your copy of OpenBLAS with a larger NUM_THREADS setting or set the environment variable OPENBLAS_NUM_THREADS to 64 or lower
BLAS : Bad memory unallocation! : 640 0x7ef44e000000
BLAS : Bad memory unallocation! : 640 0x7ef450000000
BLAS : Bad memory unallocation! : 640 0x7ef3d2000000
BLAS : Bad memory unallocation! : 640 0x7ef3c0000000
Segmentation fault (core dumped)
```
To prevent this you can either `export OPENBLAS_NUM_THREADS=64` or restrict the cores visible to the singularity image with the option `taskset -c 0-31`.
```bash
singularity exec -B /volatile,/path/to/lambdaml:/usr/src/lambdaml lambdaml-cu129.sif taskset -c 0-31 python3 /usr/src/lambdaml/pyscripts/<SCRIPT>.py --help
```

</details>

### Installation by Hand

Begin by cloning the repository
```bash
git clone https://github.com/mfmceneaney/lambdaml.git
```
Create and activate a virtual python environment for your project
```bash
python3 -m venv venv
source venv/bin/activate
```

Install the python modules.  These are listed in [pyproject.toml](pyproject.toml) and are all available with pip.
```bash
pip install -e .
```

Install the extra PyTorch-Geometric extensions needed for some of the GNN models.
```bash
pip install -r requirements-pyg-pt28-cpu.txt # Adjust pytorch and cuda version as needed.
```

<details>
<summary>:x: Installing PyTorch-Geometric on an HPC Cluster</summary>

Follow the installation instructions on the [PyTorch Geometric Documentation](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html).

If you are on *Ifarm* or another HPC cluster with a firewall, you will probably get an error like this:
```
WARNING: Retrying (Retry(total=1, connect=None, read=None, redirect=None, status=None)) after connection broken by 'ProtocolError('Connection aborted.', ConnectionResetError(104, 'Connection reset by peer'))': /wheels/repo.html
ERROR: Could not find a version that satisfies the requirement pyg-lib (from versions: none)
ERROR: No matching distribution found for pyg-lib
```
In this case, try downloading locally whatever distribution you need from the repo link posted on the installation page for installing with pip.  This will look like `https://data.pyg.org/whl/torch-${TORCH_VERSION}+${CUDA_VERSION}.html`.
Then transfer the downloaded distribution (e.g. with scp or rsync) to ifarm.

In your virtual environment you can now install from the local path:
```
pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f /path/to/distribution/you/just/uploaded
```

For your convenience put your packages in some directory `/path/to/packages` and use the CUDA docker file to install from this path.  You will need to mount the directory to `/pyg_packages` for the build to succeed.
```bash
docker build -v /path/to/packages:/pyg_packages -f /path/to/lambdaml/Dockerfile.cu129 -t lambdaml-project /path/to/lambdaml
```
</details>

Add the following to your startup script
```bash
cd /path/to/lambdaml
source $PWD/env/env.sh # csh version also available
cd
```

## :green_circle: Getting Started
Run the project pipelines for dataset creation, hyperparameter optimization, model selection, and model deployment in [pyscripts](pyscripts/).  From within the container, you can run:
```bash
python3 pyscripts/<some_script.py> --help
```

However, for running with actual $\Lambda$ data you will want to first produce the `REC::Kinematic` banks using [CLAS12-Trains](https://github.com/mfmceneaney/CLAS12-Trains).  Assume your output hipo files are in a folder designated by the environment variable `$C12TRAINS_OUTPUT_MC` for MC simulation and `$C12TRAINS_OUTPUT_DT` for data.

Then, you can run the python scripts for data set creation from outside the container.  You will want to mount the `$LAMBDAML_HOME` and your output directory, e.g., `export VOLATILE_DIR=/work/clas12/users/$USER/`.  For the MC simulation dataset run:
```bash
singularity exec \
-B $VOLATILE_DIR,$LAMBDAML_HOME:/usr/src/lambdaml lambdaml-cu129.sif \
python3 /usr/src/lambdaml/pyscripts/run_pipeline_preprocessing.py \
--file_list $C12TRAINS_OUTPUT_MC/*.hipo \
--banks REC::Particle REC::Kinematics MC::Lund \
--step 100000 \
--out_dataset_path $VOLATILE_DIR/src_dataset \
--lazy_ds_batch_size 100000 \
--num_workers 0 \
--log_level info
```

And similarly, for the real data (unlabelled) dataset, run:
```bash
singularity exec \
-B $VOLATILE_DIR,$LAMBDAML_HOME:/usr/src/lambdaml lambdaml-cu129.sif \
python3 /usr/src/lambdaml/pyscripts/run_pipeline_preprocessing.py \
--file_list $C12TRAINS_OUTPUT_DT/*.hipo \
--banks REC::Particle REC::Kinematics \
--step 100000 \
--out_dataset_path $VOLATILE_DIR/tgt_dataset \
--lazy_ds_batch_size 100000 \
--num_workers 0 \
--log_level info
```

You can then run the TIToK training script like so:
```bash
singularity exec \
-B $VOLATILE_DIR,$LAMBDAML_HOME:/usr/src/lambdaml lambdaml-cu129.sif \
taskset -c 0-31 \
python3 /usr/src/lambdaml/pyscripts/run_pipeline_titok.py \
--src_root $VOLATILE_DIR/src_dataset \
--tgt_root $VOLATILE_DIR/tgt_dataset \
--out_dir $VOLATILE_DIR/experiments \
--use_lazy_dataset \
--log_level info \
--batch_size 32 \
--nepochs 10
```

And you can run a hyperparameter optimization study like so:
```bash
singularity exec \
-B $VOLATILE_DIR,$LAMBDAML_HOME:/usr/src/lambdaml lambdaml-cu129.sif \
taskset -c 0-31 \
python3 /usr/src/lambdaml/pyscripts/run_optimize_titok.py \
--src_root $VOLATILE_DIR/src_dataset \
--tgt_root $VOLATILE_DIR/tgt_dataset \
--out_dir $VOLATILE_DIR/experiments \
--use_lazy_dataset \
--log_level info \
--batch_size 32 \
--nepochs 10 \
--opt__storage_url "sqlite:///$VOLATILE_DIR/experiments/optuna_study.db" \
--opt__suggestion_rules 'lr=float:0.0001:0.01:log' \
'num_layers_gnn=int:3:8' \
'alpha_fn=cat:0.1,0.01,sigmoid_growth,sigmoid_decay,linear_growth,linear_decay'
```

#

Contact: matthew.mceneaney@duke.edu
