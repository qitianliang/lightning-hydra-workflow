import os
import sys  # 導入 sys 以便退出
from pathlib import Path
from typing import List

from omegaconf import DictConfig

# 導入我們的日誌記錄器
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class CommandBuilder:
    """一個專門負責構建和格式化 shell 指令的服務類。 它將指令生成的複雜性從業務邏輯 (Tasks) 中解耦出來。"""

    def __init__(self, cfg: DictConfig):
        """初始化指令構建器，並立即獲取和驗證必要的環境變數。"""
        self.cfg = cfg
        self.conda_env = self.cfg.sweep_task.conda_env
        self.wandb_project = self.cfg.wandb.project
        self.wandb_entity = self.cfg.wandb.entity

        # 【關鍵修改】: 初始化時獲取並存儲 W&B 環境變數
        self.wandb_env_prefix = self._initialize_wandb_env()
        self.cmd = self._init_conda_cmd()

    def _init_conda_cmd(self) -> str:
        if os.sep in self.conda_env:
            cmd = "-p"
        else:
            cmd = "-n"
        return cmd

    def _initialize_wandb_env(self) -> str:
        """從環境中讀取 W&B 變數，如果缺失則失敗退出。"""
        log.info("Reading WANDB_API_KEY and WANDB_BASE_URL from environment...")
        api_key = os.getenv("WANDB_API_KEY")
        base_url = os.getenv("WANDB_BASE_URL")

        if not api_key or not base_url:
            log.error("Missing required environment variables for W&B Local.")
            log.error("Please set both 'WANDB_API_KEY' and 'WANDB_BASE_URL' in your shell.")
            log.error(
                "Example:\n  export WANDB_API_KEY='your_local_key'\n  export WANDB_BASE_URL='http://your.server.ip:port'"
            )
            sys.exit(1)

        log.info("✅ W&B environment variables found.")
        # 使用單引號包裹值，以處理 key 中可能存在的特殊字符
        return f"WANDB_API_KEY='{api_key}' WANDB_BASE_URL='{base_url}'"

    def build_wandb_sweep_command(self) -> list:
        """構建用於創建 W&B Sweep 的指令列表。"""
        # 注意：wandb sweep 命令本身不需要注入環境變數，
        # 因為它由主腳本直接運行，可以繼承環境。
        # 但為了統一和健壯，我們也可以使用 conda run 來執行。
        sweep_config_path = self.cfg.sweep_task.config_path

        command = [
            "conda",
            "run",
            self.cmd,
            self.conda_env,
            "--no-capture-output",
            "wandb",
            "sweep",
            sweep_config_path,
            "--project",
            self.wandb_project,
            "--entity",
            self.wandb_entity,
        ]
        return command

    def build_agent_worker_command(self, sweep_path: str) -> str:
        """構建單個 W&B Agent 工人的 bash 指令。"""
        max_failures = self.cfg.sweep_task.get("agent_max_initial_failures", 5)

        agent_cmd_str = (
            f"conda run {self.cmd} {self.conda_env} --no-capture-output wandb agent {sweep_path}"
        )

        # 【關鍵修改】: 自動注入 W&B 環境變數
        full_worker_command_str = (
            f"set -e; "
            f"{self.wandb_env_prefix} WANDB_AGENT_MAX_INITIAL_FAILURES={max_failures} {agent_cmd_str}; "
            f"exit"
        )
        return full_worker_command_str

    def build_evaluation_worker_script(
        self, commands: List[str], device: str, eval_session_name: str
    ) -> str:
        """【關鍵重構】: 為評估工人生成一個極簡的、基於 '&&' 的指令鏈腳本。"""
        worker_log_dir = Path(f"logs/workflow/evaluate/{eval_session_name}")
        worker_log_dir.mkdir(parents=True, exist_ok=True)
        worker_log_file = worker_log_dir / f"gpu_{device}.log"

        # 將每個指令用括號包起來，確保其原子性，然後用 '&&' 連接
        # `&& \` 可以在 shell 中換行，增加可讀性
        chained_command = " && \\\n".join([f"({cmd})" for cmd in commands])

        # 生成一個只包含最核心邏輯的極簡腳本
        worker_script = f"""
# 1. 確保日誌目錄存在並重定向所有輸出
mkdir -p '{worker_log_dir}'
# 2. 设置tee命令，同时输出到终端和日志文件
exec > >(tee '{worker_log_file}') 2>&1

# 3. 设置 -e，任何指令失败则立即退出
set -e

# 4. 打印启动信息
echo "Worker for GPU {device} started at $(date). Logging to {worker_log_file}"
echo "Executing {len(commands)} tasks sequentially..."
echo "---"

# 5. 直接执行指令链
{chained_command}

echo "---"
echo "✅ All tasks for GPU {device} finished successfully at $(date)."
# 6. 确保退出
exit
"""
        return worker_script

    def build_training_run_command(self, overrides: list, seed: int, group_name: str) -> str:
        """為單次的多種子評估實驗構建訓練指令。"""
        # 複製基礎指令列表以避免修改原始配置
        base_args = list(self.cfg.evaluate_task.run_command.base_args)

        # 【關鍵修正 1】: 不再手動添加 'python -u'，而是在已有的 'python' 後面插入 '-u'
        # 這確保了指令的正確性，並開啟了無緩衝日誌輸出
        if base_args and base_args[0] == "python":
            base_args.insert(1, "-u")

        train_cmd_parts = (
            base_args + overrides + [f"seed={seed}", f"logger.wandb.group={group_name}"]
        )

        # 直接拼接成最終的 python 指令
        final_command = " ".join(train_cmd_parts)
        log.info(f"{self.conda_env=}")
        return f"{self.wandb_env_prefix} conda run {self.cmd} {self.conda_env} --no-capture-output {final_command}"
