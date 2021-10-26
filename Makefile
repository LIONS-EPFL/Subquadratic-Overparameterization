push:
	rsync -avu ./src ./environment.yml ./hpc_gpu.sh ./hpc_cpu.sh ./teacher_model.pth runner.py $(SERVER_USERNAME)@$(SERVER_NAME):~/overparameterization

pull:
	if [ "$$SERVER_NAME" ]; then rsync -av $(SERVER_USERNAME)@$(SERVER_NAME):~/overparameterization/output ./; fi

.PHONY: pull
