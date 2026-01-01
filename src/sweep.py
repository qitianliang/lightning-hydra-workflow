import json
import os
import re
import subprocess  # nosec B404, B603
import sys
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import hydra
import pandas as pd
import yaml
from omegaconf import DictConfig, OmegaConf
from rootutils import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from wandb.apis.public import Run, Sweep

# 導入我們所有的服務和工具
from src.services.command_builder import CommandBuilder
from src.services.tmux_service import TmuxService
from src.services.wandb_service import WandbService
from src.utils import (
    RankedLogger,
    send_email_with_dataframe,
    send_smtp_email,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)


# =====================================================================================
# 評估策略類 (Evaluation Strategies)
# =====================================================================================
class EvaluationStrategy(ABC):
    """評估策略的抽象基類。"""

    def __init__(
        self,
        wandb_service: WandbService,
        tmux_service: TmuxService,
        command_builder: CommandBuilder,
        cfg: DictConfig,
        group_name: str,
    ):
        self.wandb_service = wandb_service
        self.tmux_service = tmux_service
        self.command_builder = command_builder
        self.cfg = cfg
        self.group_name = group_name

    @abstractmethod
    def execute(self, best_run: Run, config_overrides: list) -> str | None:
        pass


class RerunStrategy(EvaluationStrategy):
    """強制重跑策略：刪除舊的 runs，然後啟動新的 runs。"""

    def execute(self, config_overrides: list) -> str | None:
        log.info(
            f"Executing RerunStrategy: Deleting existing runs in group '{self.group_name}'..."
        )
        self.wandb_service.delete_runs_in_group(self.group_name)

        eval_session_name = self.group_name
        if self.tmux_service.session_exists(eval_session_name):
            log.error(f"tmux session '{eval_session_name}' already exists.")
            sys.exit(1)

        num_seeds = self.cfg.evaluate_task.num_seeds
        devices = self.cfg.sweep_task.devices

        # 【重構亮點】: 複雜的指令生成邏輯被簡化為幾行清晰的調用
        # 1. 為每個種子生成訓練指令
        all_commands = [
            self.command_builder.build_training_run_command(
                overrides=config_overrides,
                seed=self.cfg.evaluate_task.seed_start + i,
                group_name=self.group_name,
            )
            for i in range(num_seeds)
        ]

        # 2. 將指令分配給每個工人
        commands_per_device = [[] for _ in devices]
        for i, command in enumerate(all_commands):
            log.info(f"{i=} {command=}")
            commands_per_device[i % len(devices)].append(command)

        # 3. 為每個工人生成其專屬的、包含任務隊列的 bash 腳本
        worker_defs = []
        for i, device in enumerate(devices):
            worker_commands = commands_per_device[i]
            if not worker_commands:
                continue

            worker_script = self.command_builder.build_evaluation_worker_script(
                worker_commands, device, eval_session_name=eval_session_name
            )
            full_worker_command = f"CUDA_VISIBLE_DEVICES={device} bash -c '{worker_script}'"
            worker_defs.append({"device": device, "command": full_worker_command})

        log.info(f"🚀 Launching {len(worker_defs)} workers to handle {num_seeds} evaluation runs.")
        self.tmux_service.create_workers_session(eval_session_name, worker_defs)
        return eval_session_name


class ResumeStrategy(EvaluationStrategy):
    """恢復策略：如果 runs 已存在，則跳過運行。"""

    def execute(self, config_overrides: list) -> str | None:
        log.info(
            f"Executing ResumeStrategy: Checking for existing runs in group '{self.group_name}'..."
        )
        existing_runs = self.wandb_service.get_runs_by_group(self.group_name)
        if existing_runs:
            log.info(f"✅ Found {len(existing_runs)} existing runs. Skipping execution.")
            return None
        else:
            log.warning("No existing runs found. Nothing to resume.")
            log.info(
                "If you want to run the evaluation, please change 'evaluate_task.mode' to 'rerun' in your config."
            )
            return None  # ... (此部分邏輯不變)
        return None


# =====================================================================================
# 任務邏輯類 (Task Logic Classes)
# =====================================================================================


class SweepTask:
    """封裝了啟動 Sweep 流程的所有業務邏輯。"""

    def __init__(
        self,
        cfg: DictConfig,
        tmux_service: TmuxService,
        command_builder: CommandBuilder,
        subprocess_module=subprocess,
    ):
        self.cfg = cfg.workflow
        self.tmux_service = tmux_service
        self.command_builder = command_builder
        self.subprocess = subprocess_module

    def run(self) -> str:  # 修改：返回 sweep_id
        """執行 Sweep 任務，並阻塞式等待其完成。"""
        log.info("▶️ Stage 1: Launching Sweep Task...")
        command = self.command_builder.build_wandb_sweep_command()

        try:
            result = self.subprocess.run(command, capture_output=True, text=True, check=True)
            full_output = result.stdout + "\n" + result.stderr
            match = re.search(r"wandb agent ([\S]+)", full_output)
            if not match:
                raise RuntimeError("Could not parse Sweep Path from wandb output.")
            full_sweep_path = match.group(1)
            sweep_id = full_sweep_path.split("/")[-1]
            log.info(f"✅ Sweep created successfully! ID: {sweep_id}")
        except Exception as e:
            log.error(f"Error creating sweep: {e}")
            sys.exit(1)

        session_name = f"{self.cfg.general.tmux_session_name}_{sweep_id}"
        if self.tmux_service.session_exists(session_name):
            log.error(f"tmux session '{session_name}' already exists.")
            sys.exit(1)

        # 啟動 Agent
        worker_defs = []
        for device in self.cfg.sweep_task.devices:
            agent_command_script = self.command_builder.build_agent_worker_command(full_sweep_path)
            full_worker_command = f"CUDA_VISIBLE_DEVICES={device} bash -c '{agent_command_script}'"
            worker_defs.append({"device": device, "command": full_worker_command})

        self.tmux_service.create_workers_session(session_name, worker_defs)

        # 更新狀態文件 (保留 latest.yaml 供一眼查看信息)
        status_dir = Path(self.cfg.general.status_dir)
        status_dir.mkdir(parents=True, exist_ok=True)
        status_file_path = status_dir / f"status_{sweep_id}.yaml"
        status = {
            "sweep_id": sweep_id,
            "sweep_path": full_sweep_path,
            "tmux_session_name": session_name,
            "sweep_config": self.cfg.sweep_task.config_path,
        }
        with open(status_file_path, "w") as f:
            yaml.dump(status, f)

        latest_file_path = Path(self.cfg.general.latest_status_file)
        if latest_file_path.is_symlink() or latest_file_path.exists():
            latest_file_path.unlink()
        os.symlink(status_file_path.name, latest_file_path)

        log.info(f"Monitor Sweep: tmux attach -t {session_name}")
        self.tmux_service.wait_for_session_to_close(
            session_name, self.cfg.evaluate_task.wait_interval_seconds
        )

        return sweep_id  # 【關鍵修改】


class EvaluateTask:
    """封裝了評估最優超參的所有業務邏輯。"""

    def __init__(
        self,
        cfg: DictConfig,
        wandb_service: WandbService,
        tmux_service: TmuxService,
        command_builder: CommandBuilder,
        subprocess_module=subprocess,
    ):
        self.cfg = cfg.workflow
        self.wandb_service = wandb_service
        self.tmux_service = tmux_service
        self.subprocess = subprocess_module
        self.command_builder = command_builder

    def run(self, sweep_id: Optional[str] = None):
        """執行評估任務。"""
        log.info("▶️ Stage 2: Launching Evaluation Task...")

        if sweep_id:
            log.info(f"Using active sweep ID from current workflow: {sweep_id}")
        elif self.cfg.evaluate_task.get("target_sweep_id"):
            sweep_id = self.cfg.evaluate_task.target_sweep_id
            log.info(f"Using target sweep ID from config: {sweep_id}")
        else:
            latest_status_file = Path(self.cfg.general.latest_status_file)
            if not latest_status_file.exists():
                log.error(f"No Sweep ID provided and latest.yaml {latest_status_file} not found.")
                sys.exit(1)
            with open(latest_status_file.resolve()) as f:
                status = yaml.safe_load(f)
                sweep_id = status["sweep_id"]
            log.info(f"Using Sweep ID from latest pointer: {sweep_id}")

        log.info("--- [Phase 2.1] Waiting for Sweep to finish on W&B server ---")
        sweep = self.wandb_service.get_sweep(sweep_id)  # type: ignore
        if not sweep:
            log.error(f"Sweep {sweep_id} could not be found on W&B.")
            sys.exit(1)

        if self.cfg.evaluate_task.wait_for_sweep_finish:
            while sweep.state != "FINISHED":
                log.info(f"  -> Sweep state is currently '{sweep.state}'. Waiting...")
                time.sleep(self.cfg.evaluate_task.wait_interval_seconds)
                sweep = self.wandb_service.get_sweep(sweep_id)
            log.info("✅ Sweep has FINISHED on W&B server.")

        log.info("--- [Phase 2.2] Finding best run and extracting hyperparameters ---")
        self.description = sweep.config.get("description", "N/A")
        self.task = f"{sweep.project}.{sweep.name}"
        best_run = self.wandb_service.find_best_run(sweep, self.cfg.evaluate_task.optimized_metric)
        if not best_run:
            log.error("Could not determine the best run from sweep.")
            sys.exit(1)

        log.info(f"🏆 Best run found: {best_run.name}")

        # config_overrides = [
        #     f"{key}={value}" for key, value in best_run.config.items() if "." in key  and isinstance(value, list)
        # ]
        config_overrides = []
        for key, value in best_run.config.items():
            if "." in key:
                val_str = f'"{value}"' if isinstance(value, list) else str(value)
                config_overrides.append(f"{key}={val_str}")

        config_dict = {}
        for item in config_overrides:
            if "=" in item:
                key, value = item.split("=", 1)
                config_dict[key] = value

        # Beautify the dictionary using json.dumps
        beautified_config = json.dumps(config_dict, indent=4, ensure_ascii=False)

        self.best_run_config = beautified_config
        log.info(f"📋 Extracted Sweep Overrides: {config_overrides}")

        log.info("--- [Phase 2.3] Executing evaluation strategy ---")
        group_name = f"{self.cfg.general.tmux_session_name}_{sweep_id}-best_run_{best_run.id}"
        mode = self.cfg.evaluate_task.mode

        strategy: EvaluationStrategy
        if mode == "rerun":
            strategy = RerunStrategy(
                self.wandb_service,
                self.tmux_service,
                self.command_builder,
                self.cfg,
                group_name,
            )
        elif mode == "resume":
            strategy = ResumeStrategy(
                self.wandb_service,
                self.tmux_service,
                self.command_builder,
                self.cfg,
                group_name,
            )
        else:
            log.error(f"Unknown evaluation mode '{mode}'.")
            sys.exit(1)

        eval_session_name = strategy.execute(config_overrides)

        if eval_session_name:
            log.info("--- [Phase 2.4] Waiting for evaluation runs to complete ---")
            self.tmux_service.wait_for_session_to_close(
                eval_session_name, self.cfg.evaluate_task.wait_interval_seconds
            )

        log.info("--- [Phase 2.5] Aggregating results and generating report ---")
        self._aggregate_and_report(sweep, best_run, group_name)

    def _aggregate_and_report(self, sweep: Sweep, best_run: Run, group_name: str):
        """私有輔助函數，負責聚合結果並生成報告。"""
        log.info(f"Aggregating results for group '{group_name}'...")
        final_runs = self.wandb_service.get_runs_by_group(group_name)
        if not final_runs:
            log.error(f"No runs found in group '{group_name}' for aggregation. Nothing to report.")
            sys.exit(1)

        final_report = {
            "sweep_id": sweep.id,
            "best_run_from_sweep": best_run.name,
            "best_run_config": best_run.config,
            "evaluation_summary": {},
        }
        data = []
        for metric in self.cfg.evaluate_task.test_metrics:
            scores = [
                run.summary.get(metric)
                for run in final_runs
                if run.summary.get(metric) is not None
            ]
            if scores:
                series = pd.Series(scores)
                final_report["evaluation_summary"][metric] = {
                    "mean": series.mean(),
                    "std": series.std(),
                    "count": len(scores),
                    "values": series.tolist(),
                }
                data.append([metric, series.mean(), series.std(), series.tolist()])

        metrics_data = pd.DataFrame(data, columns=["metric", "mean", "std", "values"])

        report_path_str = self.cfg.evaluate_task.report_path.format(sweep_id=sweep.id)
        report_path = Path(report_path_str)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(
                final_report,
                f,
                indent=4,
            )

        log.info("--- Final Report ---")
        log.info(f"\n{json.dumps(final_report, indent=4)}")
        log.info(f"✅ Report saved to {report_path}")
        if metrics_data.empty:
            log.info("No metrics data available.")
        else:
            log.info("--- Metrics Data ---")
            metrics_data["mean"] = metrics_data["mean"].apply(
                lambda x: f"{x:.4g}" if pd.notnull(x) else ""
            )
            metrics_data["std"] = metrics_data["std"].apply(
                lambda x: f"{x:.4f}" if pd.notnull(x) else ""
            )

            log.info(f'\n{metrics_data[["metric", "mean", "std"]]}')
            metrics_data.to_csv(report_path.with_suffix(".csv"), index=False)
            log.info(f"✅ Metrics data saved to {report_path.with_suffix('.csv')}")
            try:
                subject = f"✅ Job Done {self.task} "
                body = f"{sweep.id=} {self.description}\n\nBest Run config:{self.best_run_config}"
                # 使用 `self.cfg` 來訪問整個工作流的配置
                send_email_with_dataframe(self.cfg.notification, subject, body, metrics_data)
            except Exception as e:
                log.error(f"An error occurred during the notification step: {e}")


# =====================================================================================
# Hydra 主入口 (Orchestrator)
# =====================================================================================
@task_wrapper
def execute(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    # 【重構亮點】: 實例化所有服務，包括新的 CommandBuilder
    log.info("Initializing services...")
    wandb_service = WandbService(cfg.workflow.wandb.entity, cfg.workflow.wandb.project)
    tmux_service = TmuxService()
    command_builder = CommandBuilder(cfg.workflow)  # CommandBuilder 需要 workflow 配置

    task_name = cfg.workflow.task_name
    active_sweep_id = None

    # --- [核心編排邏輯] ---
    # 如果是 sweep 或者是 all 模式，先啟動 sweep
    if task_name in ["sweep", "all"]:
        sweep_task = SweepTask(cfg, tmux_service, command_builder)
        active_sweep_id = sweep_task.run()

    # 如果是 evaluate 或者是 all 模式，啟動評估
    if task_name in ["evaluate", "all"]:
        evaluate_task = EvaluateTask(cfg, wandb_service, tmux_service, command_builder)
        # 傳遞 active_sweep_id。如果是獨立運行 evaluate，此處為 None，內部會走尋址邏輯
        evaluate_task.run(sweep_id=active_sweep_id)

    return {}, {}


@hydra.main(config_path="../configs", config_name="sweep.yaml", version_base=None)
def main(cfg: DictConfig):
    """主流程編排器。"""
    log.info("Workflow Orchestrator Initialized")

    log.info("Loaded Workflow Configuration:")
    config_yaml_string = OmegaConf.to_yaml(cfg.workflow)
    log.info(f"\n{config_yaml_string}")
    execute(cfg)


if __name__ == "__main__":
    main()
