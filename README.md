# Overparameterization

Code for Subquadratic Overparameterization for Shallow Neural Networks NeurIPS 2021.

## Usage

For possible arguments see:

```bash
python runner.py -h
```

Example:

```bash
source .env
python runner.py --exp-name mnist-test --d1 1000 --dataset MNIST --lr 0.01 --w1 0.03 --w2 0.04 --epochs 300 --batch-size 128 --mse-loss --remote
```

_Note that `--remote` requires the [Slurm workload manager][slurm] on the server._

## Recreating experiments

To recreate Figure 1 from the paper see `notebook.ipynb` by running the following:

```bash
source .env
jupyter notebook notebook.ipynb
```

[slurm]: https://slurm.schedmd.com/


## Installation

- CPU:  

    ```bash
    conda env create -f environment_cpu.yml
    source activate overparameterization
    ```

- GPU enabled:

    ```bash
    conda env create -f environment_gpu.yml
    source activate overparameterization
    ```


### Server install 

_Note: The server setup requires the server to use [Slurm][slurm] as its queuing manager._

1. Locally change the `.env` file to:
    
    ```bash
    PYTHONPATH=.
    export SERVER_USERNAME="myusername"
    export SERVER_NAME="myserver.com"
    ```

2. On the server create the necessary folder:
    
    ```bash
    source .env
    ssh $(SERVER_USERNAME)@$(SERVER_NAME)
    mkdir ~/overparameterization
    mkdir ~/overparameterization/output
    ```

3. Push the repository to the server:
    
    ```bash
    source .env
    make push
    ```

4. Install the python dependencies on the server:
    
    ```bash
    ssh $(SERVER_USERNAME)@$(SERVER_NAME)
    cd ~/overparameterization
    conda env create -f environment.yml
    ```
