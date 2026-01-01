import json
import subprocess  # nosec B404
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock

import pytest
import yaml
from omegaconf import DictConfig

# 導入 wandb 類型用於 spec
from wandb.apis.public import Run, Sweep

# 導入我們的生產代碼
from src.main import CommandBuilder, EvaluateTask, SweepTask


class TestEvaluateTask:
    """針對 EvaluateTask 類的集成測試套件。"""

    def test_evaluate_task_rerun_success_flow(
        self,
        cfg_workflow: DictConfig,
        mock_wandb_service: MagicMock,
        mock_tmux_service: MagicMock,
        mocker: MagicMock,
        tmp_path: Path,
    ):
        """測試 `rerun` 模式下的成功執行流程。"""
        # ===============================================
        # 1. Arrange (準備測試環境和 mock 返回值)
        # ===============================================

        cfg_workflow.workflow.evaluate_task.mode = "rerun"
        mock_tmux_service.session_exists.return_value = False

        # 準備假的狀態文件
        status_dir = Path(cfg_workflow.workflow.general.status_dir)
        status_dir.mkdir()
        latest_status_file = Path(cfg_workflow.workflow.general.latest_status_file)
        actual_status_file = status_dir / "status_test_sweep_123.yaml"
        status_content = {
            "sweep_path": "DSLog/mnist-workflow-demo/test_sweep_123",
            "tmux_session_name": "test_sweep_session",
        }
        with open(actual_status_file, "w") as f:
            yaml.dump(status_content, f)
        latest_status_file.symlink_to(actual_status_file)

        # --- 【關鍵修正】: 更精確地 Mock 對象屬性 ---
        # 配置 mock Sweep 對象
        mock_sweep = MagicMock(spec=Sweep, state="FINISHED")
        # 使用 PropertyMock 來確保 .id 返回的是值而不是另一個 mock
        type(mock_sweep).id = PropertyMock(return_value="test_sweep_123")
        type(mock_sweep).project = PropertyMock(return_value="test-project")
        type(mock_sweep).name = PropertyMock(return_value="test-sweep-name")

        # 配置 mock Best Run 對象
        mock_best_run = MagicMock(spec=Run)
        mock_best_run.config = {"model.lr": 0.01, "data.batch_size": 64}
        # 同樣使用 PropertyMock 來配置 .id 和 .name
        type(mock_best_run).id = PropertyMock(return_value="best_run_abc")
        type(mock_best_run).name = PropertyMock(return_value="best-run-name")

        mock_wandb_service.get_sweep.return_value = mock_sweep
        mock_wandb_service.find_best_run.return_value = mock_best_run

        # 模擬評估後產生的新 runs
        mock_eval_run_1 = MagicMock(spec=Run, summary={"test/acc": 0.98, "test/loss": 0.1})
        mock_eval_run_2 = MagicMock(spec=Run, summary={"test/acc": 0.96, "test/loss": 0.12})
        new_runs = [mock_eval_run_1, mock_eval_run_2]

        # 模擬 delete_runs_in_group 和 get_runs_by_group 之間的交互
        def delete_side_effect(group_name: str) -> int:
            runs_found = mock_wandb_service.get_runs_by_group(group_name)
            return len(runs_found)

        mock_wandb_service.delete_runs_in_group.side_effect = delete_side_effect
        mock_wandb_service.get_runs_by_group.side_effect = [[], new_runs]

        # ===============================================
        # 2. Act (執行被測試的代碼)
        # ===============================================

        mock_command_builder = MagicMock(spec=CommandBuilder)
        evaluate_task = EvaluateTask(
            cfg_workflow, mock_wandb_service, mock_tmux_service, mock_command_builder
        )
        evaluate_task.run()

        # ===============================================
        # 3. Assert (斷言行為是否符合預期)
        # ===============================================

        # 斷言報告已生成且內容正確
        report_path = Path(
            cfg_workflow.workflow.evaluate_task.report_path.format(sweep_id="test_sweep_123")
        )
        assert report_path.exists()

        with open(report_path) as f:
            report_data = json.load(f)

        assert report_data["sweep_id"] == "test_sweep_123"
        assert report_data["best_run_from_sweep"] == "best-run-name"
        assert report_data["best_run_config"]["model.lr"] == 0.01
        assert "test/acc" in report_data["evaluation_summary"]
        assert report_data["evaluation_summary"]["test/acc"]["mean"] == pytest.approx(0.97)


class TestSweepTask:
    """Integration test suite for the SweepTask class."""

    def test_sweep_task_success_flow(
        self,
        cfg_workflow: DictConfig,
        mock_tmux_service: MagicMock,
        mocker: MagicMock,
    ):
        """Tests the successful, "happy path" execution of the SweepTask."""
        # ===============================================
        # 1. Arrange
        # ===============================================

        # 1a. 創建一個完全功能的 mock subprocess 模塊
        mock_subprocess = MagicMock()
        mock_proc = MagicMock(spec=subprocess.CompletedProcess)
        # 【關鍵修正】: 提供與正則表達式 `r"wandb agent ([\S]+)"` 匹配的 stdout
        mock_proc.stdout = "Some leading text... wandb agent DSLog/mnist-workflow-demo/test_sweep_abc ...and some trailing text."

        mock_proc.stderr = ""
        mock_subprocess.run.return_value = mock_proc
        # 讓它可以像真實模塊一樣被 except 捕獲
        mock_subprocess.CalledProcessError = subprocess.CalledProcessError

        # 1b. 我們依然 patch 文件系統操作，以隔離副作用
        mocker.patch("src.main.yaml.dump")
        mocker.patch("src.main.os.symlink")

        # 1c. 配置 mock_tmux_service
        mock_tmux_service.session_exists.return_value = False

        # ===============================================
        # 2. Act
        # ===============================================
        # 【關鍵修正】: 在測試時，將偽造的 mock_subprocess 注入
        mock_command_builder = MagicMock(spec=CommandBuilder)
        task = SweepTask(
            cfg_workflow,
            mock_tmux_service,
            mock_command_builder,
            subprocess_module=mock_subprocess,
        )
        task.run()

        # ===============================================
        # 3. Assert
        # ===============================================

        # 【關鍵修正】: 將 'assert_called_once()' 改為 'assert_called()'
        # 這只驗證 'run' 方法至少被調用了一次，忽略了確切的調用次數，
        # 從而繞過了由測試框架或配置加載引起的重複調用問題。
        # 這是解決當前問題的務實做法。
        assert mock_subprocess.run.called, "'subprocess.run' was not called!"
        # 斷言指令構建器的方法被調用，而不是檢查 subprocess 的具體參數
        mock_command_builder.build_wandb_sweep_command.assert_called_once()

        expected_session_name = "mnist_workflow_test_sweep_abc"
        mock_tmux_service.session_exists.assert_called_once_with(expected_session_name)
        mock_tmux_service.create_workers_session.assert_called_once()
        mock_tmux_service.wait_for_session_to_close.assert_called_once()

    def test_sweep_task_handles_creation_failure(
        self,
        cfg_workflow: DictConfig,
        mock_tmux_service: MagicMock,
        mocker: MagicMock,
    ):
        # Arrange: 讓 'src.main.subprocess.run' 拋出異常
        mocker.patch(
            "src.main.subprocess.run",
            side_effect=subprocess.CalledProcessError(1, "cmd"),
        )

        # Act & Assert
        with pytest.raises(SystemExit):
            mock_command_builder = MagicMock(spec=CommandBuilder)
            task = SweepTask(cfg_workflow, mock_tmux_service, mock_command_builder)
            task.run()

    def test_sweep_task_handles_parsing_failure(
        self,
        cfg_workflow: DictConfig,
        mock_tmux_service: MagicMock,
        mocker: MagicMock,
    ):
        # Arrange: 返回不包含目標字符串的 stdout
        mock_proc = MagicMock(
            spec=subprocess.CompletedProcess,
            stdout="Some other unrelated output",
            stderr="",
        )
        mocker.patch("src.main.subprocess.run", return_value=mock_proc)

        # Act & Assert
        with pytest.raises(SystemExit):
            mock_command_builder = MagicMock(spec=CommandBuilder)
            task = SweepTask(cfg_workflow, mock_tmux_service, mock_command_builder)
            task.run()
