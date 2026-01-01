import shutil
import subprocess  # nosec B404
import time
from typing import List

from src.utils import RankedLogger

log = RankedLogger(__name__)


class TmuxService:
    """一個專門負責與 tmux 交互的服務類。 封裝了所有 tmux 子進程的調用，將業務邏輯與終端操作解耦。"""

    def session_exists(self, session_name: str) -> bool:
        """檢查指定名稱的 tmux session 是否存在。"""
        tmux_path = shutil.which("tmux")
        result = subprocess.run(
            [tmux_path, "has-session", "-t", session_name],
            capture_output=True,
            text=True,
            check=False,  # nosec B603
        )  # nosec B603, B607
        return result.returncode == 0

    def create_workers_session(self, session_name: str, commands_per_worker: List[dict]):
        """創建一個新的 tmux session，並為每個工人啟動一個窗口來順序執行指令。

        Args:
            session_name (str): 要創建的 tmux session 的名稱。
            commands_per_worker (List[dict]): 一個字典列表，每個字典包含 'device' 和 'command'。
        """
        if not commands_per_worker:
            log.info("[yellow]Warning: No worker commands provided to create session.[/yellow]")
            return

        is_first_window = True
        for worker_info in commands_per_worker:
            device = worker_info["device"]
            command = worker_info["command"]
            window_name = f"GPU-{device}"

            if is_first_window:
                tmux_path = shutil.which("tmux")
                subprocess.run(
                    [
                        tmux_path,
                        "new-session",
                        "-d",
                        "-s",
                        session_name,
                        "-n",
                        window_name,
                        command,
                    ],
                    check=False,
                )  # nosec B603, B607
                is_first_window = False
            else:
                tmux_path = shutil.which("tmux")
                subprocess.run(
                    [tmux_path, "new-window", "-t", session_name, "-n", window_name, command],
                    check=False,
                )  # nosec B603, B607

    def wait_for_session_to_close(self, session_name: str, interval_seconds: int):
        """輪詢並等待一個 tmux session 結束。

        Args:
            session_name (str): 要等待的 session 名稱。
            interval_seconds (int): 輪詢的時間間隔（秒）。
        """
        log.info(
            f"Waiting for tmux session '{session_name}' to close. Checking every {interval_seconds}s..."
        )
        while self.session_exists(session_name):
            log.info(f"  -> Session '{session_name}' is still running. Waiting...")
            time.sleep(interval_seconds)
        log.info(f"✅ Tmux session '{session_name}' finished.")

    def kill_session(self, session_name: str):
        """強制殺死一個 tmux session。"""
        log.info(f"Killing tmux session: {session_name}")
        tmux_path = shutil.which("tmux")
        subprocess.run(
            [tmux_path, "kill-session", "-t", session_name], check=False
        )  # nosec B603, B607
