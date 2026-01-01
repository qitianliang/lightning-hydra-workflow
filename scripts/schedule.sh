#!/bin/bash
PY=cu129
# shellcheck source=/dev/null
source "$(conda info --base)/etc/profile.d/conda.sh"

conda activate base # This might be redundant if the next activate works
conda activate ${PY}

# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

# python src/train.py trainer.max_epochs=5 logger=csv

# python src/train.py trainer.max_epochs=10 logger=csv

# python src/train.py experiment=example +optimized_metric=val/acc logger=wandb trainer=gpu
# python src/train.py experiment=example logger=wandb logger.wandb.offline=True trainer=gpu logger.wandb.save_dir=/llm/github/lightning-hydra-template +logger.wandb.save_code=True
# python src/train.py experiment=example logger=wandb logger.wandb.offline=True trainer=ddp trainer.max_epochs=1
python src/train.py experiment=example logger=wandb_alert logger.wandb.offline=true trainer=gpu trainer.max_epochs=1 tags="[demo,test]"
# python src/train.py experiment=example trainer=gpu
