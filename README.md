# Lightning-Hydra-Workflow

基于[lighting-hydra-template](README.template.md)

✅ BUG Fix

- 修正wandb 记录diff.patch BUG
  - 指定`logs`子目录后无法正确识别项目根目录(`configs/hydra/default.yaml`)
  - diff.patch 在运行结束后复制到实际运行logs目录(`src/utils/cleanup_context.py`)

✅ Feature Enhance

- HPO

  使用configs/workflow/mnist.yaml配置 `task_name` 支持以下参数

  - `all` : sweep+evaluate （常用）单次运行超参实验
  - `sweep` : 仅超参
  - `evaluate` :（复跑）例如修改test 阶段输出指标

  其中`evaluate`模式下 支持`mode=resume/rerun`

- Email 通知

  服务于超参调优工作流，记录最优超参对应的模型测试集指标

  包含：

  - 实验基础信息，最优超参加入email body(`src/utils/helpers.py`)
  - 支持可配置Avg metrics `configs/workflow/mnist.yaml`key:`evaluate.test_metrics`

- 日志修改为相对路径+行号，方便debug tracing(`src/utils/pylogger.py`)
