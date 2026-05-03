import json
import os
import queue
import threading
import traceback
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from tkinter import END, StringVar, Text, Tk
from tkinter import ttk

from experiment_config import (
    build_experiment_catalog,
    get_default_config_path,
    get_group_trials,
    load_project_config,
)
from experiment_control import ExperimentControlState
from main_sim_grasp_urdf import run as run_experiment


class QueueLogWriter:
    def __init__(self, ui_queue, file_path):
        self.ui_queue = ui_queue
        self.file_obj = open(file_path, "a", encoding="utf-8")

    def write(self, text):
        if not text:
            return 0
        self.file_obj.write(text)
        self.file_obj.flush()
        self.ui_queue.put(("log", text))
        return len(text)

    def flush(self):
        self.file_obj.flush()

    def close(self):
        self.file_obj.close()


class ExperimentControlPanel:
    def __init__(self, root):
        self.root = root
        self.root.title("实验轻量控制面板")
        self.root.geometry("1080x700")

        self.default_config_path = get_default_config_path()
        self.base_config = load_project_config(self.default_config_path)
        self.catalog = build_experiment_catalog(self.base_config)
        self.category_by_name = {item["name"]: item for item in self.catalog}
        self.group_by_display_name = {}

        self.worker_thread = None
        self.control_state = None
        self.log_queue = queue.Queue()
        self.log_writer = None

        self.category_var = StringVar()
        self.group_var = StringVar()
        self.trial_var = StringVar(value="1")
        self.status_var = StringVar(value="未启动")
        self.log_path_var = StringVar(value="日志文件：未生成")

        self._build_ui()
        self._init_options()
        self.root.after(120, self._poll_log_queue)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self):
        top_frame = ttk.Frame(self.root, padding=12)
        top_frame.pack(fill="x")

        ttk.Label(top_frame, text="实验类别").grid(row=0, column=0, sticky="w")
        self.category_combo = ttk.Combobox(top_frame, textvariable=self.category_var, state="readonly", width=16)
        self.category_combo.grid(row=0, column=1, padx=(8, 18), sticky="w")
        self.category_combo.bind("<<ComboboxSelected>>", self._on_category_changed)

        ttk.Label(top_frame, text="实验组").grid(row=0, column=2, sticky="w")
        self.group_combo = ttk.Combobox(top_frame, textvariable=self.group_var, state="readonly", width=38)
        self.group_combo.grid(row=0, column=3, padx=(8, 18), sticky="ew")
        self.group_combo.bind("<<ComboboxSelected>>", self._on_group_changed)

        ttk.Label(top_frame, text="起始次数").grid(row=0, column=4, sticky="w")
        self.trial_spinbox = ttk.Spinbox(top_frame, from_=1, to=1, textvariable=self.trial_var, width=8)
        self.trial_spinbox.grid(row=0, column=5, padx=(8, 18), sticky="w")

        self.start_button = ttk.Button(top_frame, text="启动", command=self.start_or_resume)
        self.start_button.grid(row=0, column=6, padx=4)

        self.pause_button = ttk.Button(top_frame, text="暂停", command=self.pause_or_resume)
        self.pause_button.grid(row=0, column=7, padx=4)

        self.reset_button = ttk.Button(top_frame, text="重置", command=self.reset_experiment)
        self.reset_button.grid(row=0, column=8, padx=4)

        top_frame.columnconfigure(3, weight=1)

        status_frame = ttk.Frame(self.root, padding=(12, 0, 12, 8))
        status_frame.pack(fill="x")
        ttk.Label(status_frame, text="状态：").pack(side="left")
        ttk.Label(status_frame, textvariable=self.status_var).pack(side="left")
        ttk.Label(status_frame, text="    ").pack(side="left")
        ttk.Label(status_frame, textvariable=self.log_path_var).pack(side="left")

        log_frame = ttk.Frame(self.root, padding=12)
        log_frame.pack(fill="both", expand=True)
        self.log_text = Text(log_frame, wrap="word", height=34)
        self.log_text.pack(side="left", fill="both", expand=True)
        scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        scrollbar.pack(side="right", fill="y")
        self.log_text.configure(yscrollcommand=scrollbar.set)

    def _init_options(self):
        category_names = [item["name"] for item in self.catalog]
        self.category_combo["values"] = category_names
        if category_names:
            self.category_var.set(category_names[0])
            self._refresh_groups()

    def _on_category_changed(self, _event=None):
        self._refresh_groups()

    def _on_group_changed(self, _event=None):
        self._refresh_trial_range()

    def _refresh_groups(self):
        category = self.category_by_name.get(self.category_var.get())
        if not category:
            self.group_combo["values"] = []
            self.group_var.set("")
            self._refresh_trial_range()
            return

        groups = category["groups"]
        values = [group["display_name"] for group in groups]
        self.group_by_display_name = {group["display_name"]: group for group in groups}
        self.group_combo["values"] = values
        if values:
            self.group_var.set(values[0])
        self._refresh_trial_range()

    def _refresh_trial_range(self):
        group = self._selected_group()
        max_trials = 1 if group is None else max(1, int(get_group_trials(group)))
        self.trial_spinbox.configure(from_=1, to=max_trials)
        try:
            trial_number = int(self.trial_var.get())
        except ValueError:
            trial_number = 1
        trial_number = max(1, min(trial_number, max_trials))
        self.trial_var.set(str(trial_number))

    def _selected_group(self):
        return self.group_by_display_name.get(self.group_var.get())

    def _selected_trial_index(self):
        group = self._selected_group()
        max_trials = 1 if group is None else max(1, int(get_group_trials(group)))
        try:
            trial_number = int(self.trial_var.get())
        except ValueError:
            trial_number = 1
        trial_number = max(1, min(trial_number, max_trials))
        self.trial_var.set(str(trial_number))
        return trial_number - 1

    def _make_launch_config(self, group):
        config = load_project_config(self.default_config_path)
        config["runtime"]["auto_run"] = True
        config["resume"]["resume_from_group_index"] = int(group["global_index"])
        config["resume"]["resume_from_trial_index"] = int(self._selected_trial_index())
        config["resume"]["restart_from_resume_group"] = True

        config_dir = os.path.join(config["tracking"]["run_root_dir"], "control_panel_configs")
        os.makedirs(config_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        config_path = os.path.join(config_dir, f"panel_launch_{timestamp}.json")
        with open(config_path, "w", encoding="utf-8") as file_obj:
            json.dump(config, file_obj, ensure_ascii=False, indent=2)
        return config_path

    def _append_ui_log(self, text):
        self.log_text.insert(END, text)
        self.log_text.see(END)

    def _worker_entry(self, config_path, log_path):
        self.log_writer = QueueLogWriter(self.log_queue, log_path)
        try:
            with redirect_stdout(self.log_writer), redirect_stderr(self.log_writer):
                run_experiment(config_path=config_path, control_state=self.control_state)
        except Exception:
            self.log_writer.write(traceback.format_exc())
        finally:
            if self.log_writer is not None:
                self.log_writer.close()
                self.log_writer = None
            self.log_queue.put(("finished", None))

    def _is_running(self):
        return self.worker_thread is not None and self.worker_thread.is_alive()

    def start_or_resume(self):
        group = self._selected_group()
        if group is None:
            self.status_var.set("请选择实验组")
            return

        if self._is_running():
            if self.control_state and self.control_state.is_paused():
                self.control_state.resume()
                self.pause_button.config(text="暂停")
                self.status_var.set(
                    f"已继续：{group['display_name']} 第 {self._selected_trial_index() + 1} 次"
                )
            return

        config_path = self._make_launch_config(group)
        logs_dir = os.path.join(self.base_config["tracking"]["run_root_dir"], "control_panel_logs")
        os.makedirs(logs_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        log_path = os.path.abspath(os.path.join(logs_dir, f"panel_{timestamp}.log"))
        self.log_path_var.set(f"日志文件：{log_path}")
        self.control_state = ExperimentControlState()
        self.worker_thread = threading.Thread(
            target=self._worker_entry,
            args=(config_path, log_path),
            daemon=True,
        )
        self.worker_thread.start()
        self.pause_button.config(text="暂停")
        self.status_var.set(
            f"运行中：{group['display_name']} 第 {self._selected_trial_index() + 1} 次"
        )

    def pause_or_resume(self):
        if not self._is_running() or self.control_state is None:
            return
        paused = self.control_state.toggle_pause()
        if paused:
            self.pause_button.config(text="继续")
            self.status_var.set("已暂停")
        else:
            self.pause_button.config(text="暂停")
            self.status_var.set("运行中")

    def reset_experiment(self):
        if not self._is_running() or self.control_state is None:
            return
        group = self._selected_group()
        target_group_index = None if group is None else int(group["global_index"])
        target_trial_index = self._selected_trial_index()
        self.control_state.request_reset(group_index=target_group_index, trial_index=target_trial_index)
        if group is None:
            self.status_var.set("已请求重置当前实验")
        else:
            self.status_var.set(
                f"已请求重置到：{group['display_name']} 第 {target_trial_index + 1} 次"
            )

    def _poll_log_queue(self):
        while True:
            try:
                event_type, payload = self.log_queue.get_nowait()
            except queue.Empty:
                break
            if event_type == "log":
                self._append_ui_log(payload)
            elif event_type == "finished":
                self.status_var.set("运行结束")
                self.pause_button.config(text="暂停")
        self.root.after(120, self._poll_log_queue)

    def _on_close(self):
        self.root.destroy()


def main():
    root = Tk()
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except Exception:
        pass
    ExperimentControlPanel(root)
    root.mainloop()


if __name__ == "__main__":
    main()
