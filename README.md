# $\Lambda$ ML

This is a project to train Graph Neural Networks (GNNs) for $\Lambda$ hyperon event identification at [CLAS12](https://www.jlab.org/physics/hall-b/clas12).
[Domain Adversarial](https://arxiv.org/abs/1505.07818), [Contrastive Adaptation](http://arxiv.org/abs/1901.00976), and [TIToK](https://www.sciencedirect.com/science/article/pii/S0893608023002137) approaches are implemented for adapting models to target data domains.

## Prerequisites

* python>=3.9

## Installation

You can use either a container image via Docker or install manually assuming you have all the prerequisites.

## Installation Via Docker

You may also install and run the project as a [Docker](https://www.docker.com) container.

Begin as above by cloning the repository:
```bash
git clone https://github.com/mfmceneaney/lambdaml.git
```

Then build the project with:
```bash
docker build -t lambdaml-project /path/to/lambdaml
```
and run it with:
```bash
docker run --rm -it lambdaml-project
```
The `--rm` option tells docker to remove the container and its data once it is shut down.
To retain the container data though, you can mount a local directory (`src`) to a directory (`dst`)
inside the container with the following:
```bash
docker run --rm -it -v <src>:<dst> lambdaml-project
```
Once you start the container you should have the following environment variables:
- `LAMBDAML_HOME`
- `LAMBDAML_APP_DIR`

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

Add the following to your startup script
```bash
cd /path/to/lambdaml
source $PWD/bin/env.sh # csh version also available
cd
```

### Installing PyTorch-Geometric

Follow the installation instructions on the [PyTorch Geometric Documentation](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html).

If you are on *Ifarm* or another HPC cluster with a firewall, you will probably get an error like this:
```
WARNING: Retrying (Retry(total=1, connect=None, read=None, redirect=None, status=None)) after connection broken by 'ProtocolError('Connection aborted.', ConnectionResetError(104, 'Connection reset by peer'))': /wheels/repo.html
ERROR: Could not find a version that satisfies the requirement pyg-lib (from versions: none)
ERROR: No matching distribution found for pyg-lib
```
In this case, try downloading locally whatever distribution you want from the repo link posted on the installation page for installing with pip.  This will look like `https://data.pyg.org/whl/torch-${TORCH_VERSION}.html`.
Then transfer the downloaded distribution (e.g. with scp or rsync) to ifarm.

In your virtual environment you can now install from the local path:
```
pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f /path/to/distribution/you/just/uploaded
```

## Getting Started
Run the project pipelines for dataset creation, hyperparameter optimization, model selection, and model deployment in [pyscripts](pyscripts/).

#

Contact: matthew.mceneaney@duke.edu
