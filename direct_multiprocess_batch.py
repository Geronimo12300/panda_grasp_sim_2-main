import argparse
import copy
import json
import os
import subprocess
import sys
import time
from datetime import datetime

from experiment_config import build_experiment_groups, get_default_config_path, load_project_config


def sanitize_name(value):
    keep = []
    for ch in value:
        if ch.isalnum() or ch in ("-", "_"):
            keep.append(ch)
        elif ch in (" ", "：", ":"):
            keep.append("_")
    text = "".join(keep).strip("_")
    return text or "group"


def make_group_specific_config(base_config, group, group_output_dir):
    config = copy.deepcopy(base_config)
    category = {
        "id": group["category_id"],
        "name": group["category_name"],
        "size_mode": group["size_mode"],
    }
    report_kind = group["report_kind"]
    standard_templates = []
    special_templates = []
    template = {
        "template_id": group["template_id"],
        "name": group["name"],
        "shapes": list(group["shapes"]),
        "report_kind": report_kind,
    }
    if "structure_mode" in group:
        template["structure_mode"] = group["structure_mode"]
    if "trials" in group:
        template["trials"] = int(group["trials"])

    if report_kind == "special":
        special_templates = [template]
    else:
        standard_templates = [template]

    config["runtime"]["auto_run"] = True
    config["runtime"]["connection_mode"] = "DIRECT"
    config["runtime"]["simulation_sleep_seconds"] = 0.0
    config["resume"]["resume_from_group_index"] = 0
    config["resume"]["resume_from_trial_index"] = 0
    config["resume"]["restart_from_resume_group"] = True
    config["paths"]["results_markdown_path"] = os.path.join(group_output_dir, "experiment_results.md")
    config["paths"]["special_results_markdown_path"] = os.path.join(group_output_dir, "special_experiment_results.md")
    config["paths"]["screenshot_dir"] = os.path.join(group_output_dir, "screens_standard")
    config["paths"]["special_screenshot_dir"] = os.path.join(group_output_dir, "screens_special")
    config["paths"]["final_render_dir"] = os.path.join(group_output_dir, "final_render")
    config["tracking"]["run_root_dir"] = os.path.join(group_output_dir, "runs")
    config["reproducibility"]["base_random_seed"] = int(config["reproducibility"]["base_random_seed"]) + group["global_index"] * 10000
    config["experiments"]["categories"] = [category]
    config["experiments"]["standard_group_templates"] = standard_templates
    config["experiments"]["special_group_templates"] = special_templates
    return config


def launch_processes(base_config_path, category_id):
    project_root = os.path.dirname(os.path.abspath(__file__))
    base_config = load_project_config(base_config_path)
    _, _, all_groups = build_experiment_groups(base_config)
    target_groups = [group for group in all_groups if group["category_id"] == category_id]
    if not target_groups:
        raise ValueError(f"未找到类别 {category_id} 对应的实验组")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    batch_root = os.path.abspath(os.path.join(project_root, "runs", f"direct_batch_{category_id}_{timestamp}"))
    config_dir = os.path.join(batch_root, "configs")
    log_dir = os.path.join(batch_root, "logs")
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    manifest = {
        "batch_root": batch_root,
        "category_id": category_id,
        "started_at": datetime.now().astimezone().isoformat(),
        "processes": [],
    }

    processes = []
    for group in target_groups:
        group_slug = f"{group['group_no']:02d}_{sanitize_name(group['display_name'])}"
        group_output_dir = os.path.join(batch_root, group_slug)
        os.makedirs(group_output_dir, exist_ok=True)
        os.makedirs(os.path.join(group_output_dir, "runs"), exist_ok=True)

        config = make_group_specific_config(base_config, group, group_output_dir)
        config_path = os.path.join(config_dir, f"{group_slug}.json")
        log_path = os.path.join(log_dir, f"{group_slug}.log")
        with open(config_path, "w", encoding="utf-8") as file_obj:
            json.dump(config, file_obj, ensure_ascii=False, indent=2)

        log_file = open(log_path, "w", encoding="utf-8")
        process = subprocess.Popen(
            [sys.executable, "main_sim_grasp_urdf.py", "--config", config_path],
            cwd=project_root,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        processes.append((group, process, log_file, log_path, config_path, group_output_dir))
        manifest["processes"].append(
            {
                "group_no": group["group_no"],
                "group_name": group["display_name"],
                "pid": process.pid,
                "config_path": config_path,
                "log_path": log_path,
                "output_dir": group_output_dir,
            }
        )
        print(f"已启动 {group['display_name']} -> PID={process.pid}")
        print(f"  配置: {config_path}")
        print(f"  日志: {log_path}")

    manifest_path = os.path.join(batch_root, "batch_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as file_obj:
        json.dump(manifest, file_obj, ensure_ascii=False, indent=2)

    print(f"\n批量测试目录：{batch_root}")
    print(f"批量清单：{manifest_path}")
    print("正在等待所有 DIRECT 进程完成...\n")

    exit_codes = {}
    try:
        while processes:
            remaining = []
            for group, process, log_file, log_path, config_path, group_output_dir in processes:
                code = process.poll()
                if code is None:
                    remaining.append((group, process, log_file, log_path, config_path, group_output_dir))
                    continue
                exit_codes[group["display_name"]] = code
                log_file.close()
                status_text = "成功" if code == 0 else f"失败(退出码={code})"
                print(f"进程结束: {group['display_name']} -> {status_text}")
                print(f"  日志: {log_path}")
            processes = remaining
            if processes:
                time.sleep(2.0)
    finally:
        for _, process, log_file, _, _, _ in processes:
            if process.poll() is None:
                process.terminate()
            log_file.close()

    summary = {
        "finished_at": datetime.now().astimezone().isoformat(),
        "exit_codes": exit_codes,
    }
    with open(os.path.join(batch_root, "batch_summary.json"), "w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, ensure_ascii=False, indent=2)

    failed = [name for name, code in exit_codes.items() if code != 0]
    if failed:
        print("\n以下组运行失败：")
        for name in failed:
            print(f"- {name}")
        return 1

    print("\n固定尺寸六组 DIRECT 多进程批量测试已完成。")
    return 0


def main():
    parser = argparse.ArgumentParser(description="DIRECT 多进程批量测试入口")
    parser.add_argument(
        "--config",
        default=get_default_config_path(),
        help="基础 JSON 配置文件路径",
    )
    parser.add_argument(
        "--category-id",
        default="fixed_size",
        help="要并行运行的实验类别，默认 fixed_size",
    )
    args = parser.parse_args()
    raise SystemExit(launch_processes(args.config, args.category_id))


if __name__ == "__main__":
    main()
