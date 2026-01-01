#!/bin/bash

# shellcheck source=/dev/null
source "$(conda info --base)/etc/profile.d/conda.sh"
pyenv="${pyenv:-cu129}"
devices="${devices:-[0,1]}"
conda activate base # This might be redundant if the next activate works
conda activate "${pyenv}"

wandb login
# sweep and evaluate
# python src/sweep.py workflow.task_name=all workflow.sweep_task.conda_env="${pyenv}" workflow.sweep_task.devices="${devices}"

# sweep
# python src/sweep.py workflow.task_name=sweep workflow.sweep_task.conda_env="${pyenv}" workflow.sweep_task.devices="${devices}"

# evaluate Default sweep logs: `logs/sweeps/status_{sweep_id}.yaml`, also can be found on wandb UI sweeps
# mode = resume, sweep_id = `0294pwn2` as example
python src/sweep.py workflow.task_name=evaluate \
    workflow.evaluate_task.mode=resume  \
    workflow.evaluate_task.target_sweep_id=0294pwn2 \
    workflow.sweep_task.conda_env="${pyenv}" \
    workflow.sweep_task.devices="${devices}"
# mode = rerun
# python src/sweep.py workflow.task_name=evaluate workflow.evaluate_task.target_sweep_id=sweep_id workflow.evaluate_task.mode=rerun workflow.sweep_task.conda_env="${pyenv}" workflow.sweep_task.devices="${devices}"
