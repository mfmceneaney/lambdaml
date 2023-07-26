# $\Lambda$ GNNs

Graph Neural Networks (GNNs) for $\Lambda$ hyperon identification at [CLAS12](https://www.jlab.org/physics/hall-b/clas12).

Required python modules (all available with pip) are listed in [requirements.txt](requirements.txt).
* Examples of creating datasets, training a model and running hyperparamter optimization are provided in [tutorials][tutorials/].
* Notebook versions are in [tutorials/ipynb][tutorials/ipynb].

## Creating a CUDA Virtual Environment

On Ifarm, try installing a new python virtual environment with:
```
/apps/python/3.9.5/bin/python3.9 -m venv venv_cuda
source /full/path/to/venv_cuda/bin/activate
which python
deactivate
```

Also a good idea to put your venv packages first in your python path with:
```
export PYTHONPATH=/full/path/to/venv_cuda/lib/python*/site-packages/:$PYTHONPATH
```

Start an interactive GPU job (not entirely sure if this is necessary but it’s nice for checking your CUDA version and verifying package installation):

```
srun -p gpu -c 2 --gres=gpu:1 --mem-per-cpu=8G --pty bash
```

Start your python3 cuda virtual environment:
```
source /full/path/to/venv_cuda/bin/activate
```

## Pytorch

Follow the instructions at [PyTorch Get Started Locallly](https://pytorch.org/get-started/locally/).
You should execute something like this:
```
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
```

## PyTorch-Geometric

Follow the installation instructions on the [PyTorch Geometric Documentation](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html).

If you are on *Ifarm*, you will probably get an error like this:
```
WARNING: Retrying (Retry(total=1, connect=None, read=None, redirect=None, status=None)) after connection broken by 'ProtocolError('Connection aborted.', ConnectionResetError(104, 'Connection reset by peer'))': /wheels/repo.html
ERROR: Could not find a version that satisfies the requirement pyg-lib (from versions: none)
ERROR: No matching distribution found for pyg-lib
```
In this case, try downloading locally whatever distribution you want from the repo link posted on the installation page for installing with pip.  This probably looks like `https://data.pyg.org/whl/torch-${TORCH_VERSION}.html`.
Then transfer the downloaded distribution (e.g. with scp) to ifarm.

In your virtual environment you can now install from the local path:
```
pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f /path/to/distribution/you/just/uploaded
```

#

Contact: matthew.mceneaney@duke.edu
