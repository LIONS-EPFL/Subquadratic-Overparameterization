import os
import sys

import subprocess

from src.model import Runner


def hpc_wrap(cmd, enable_gpu=False):
    """Takes a python script and wraps it in `sbatch` over `ssh`.

    :param cmd: The python script to be executed.
    :param enable_gpu:
    :return: Return array that can be executed with `subprocess.call`.
    """
    python_cmd_args = " ".join(map(lambda x: "'{}'".format(x), cmd))

    if enable_gpu:
        bash_script = 'hpc_gpu.sh'
    else:
        bash_script = 'hpc_cpu.sh'

    server_cmd = "cd overparameterization; sbatch {} {}".format(bash_script, python_cmd_args)
    if os.environ.get('SERVER_NAME') is None:
        raise RuntimeError('Environment variables `SERVER_NAME` and `SERVER_USERNAME` need to be specified for remote execution.')
    else:
        ssh_cmd = ["ssh", os.environ['SERVER_USERNAME'] + '@' + os.environ['SERVER_NAME'], server_cmd]
        return ssh_cmd


def server_execute(cmd, enable_gpu=False):
    """Executes a script over `ssh` using the Slurm queuing system.

    :param cmd:
    :param enable_gpu:
    :return:
    """
    ssh_cmd = hpc_wrap(cmd, enable_gpu=enable_gpu)
    print(ssh_cmd)
    print(subprocess.check_output(ssh_cmd))


if __name__ == '__main__':
    args = Runner.from_parser()
    cmd_list = sys.argv
    runner = Runner(args)

    if args.remote:
        while '--remote' in cmd_list:
            cmd_list.remove('--remote')
        server_execute(['python'] + cmd_list, enable_gpu=args.cuda)

    elif args.dummy_run:
        pass

    else:
        runner.run()
