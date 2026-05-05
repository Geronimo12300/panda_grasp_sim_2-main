import os
import re
import shutil

from experiment_config import get_group_screenshot_dir, get_group_trials


SUCCESS_LABELS = {"成功", "鎴愬姛"}
PLANNING_OK_LABELS = {"符合", "绗﹀悎"}


def ensure_output_paths(screenshot_dir, special_screenshot_dir, render_dir):
    os.makedirs(screenshot_dir, exist_ok=True)
    os.makedirs(special_screenshot_dir, exist_ok=True)
    os.makedirs(render_dir, exist_ok=True)


def render_group_links(records):
    if not records:
        return ["- 暂无截图"]

    lines = []
    for record in records:
        links = record["links"]
        lines.append(
            f"- 第{record['trial']}次："
            f"[顶视图]({links['top']}) / "
            f"[左视图]({links['left']}) / "
            f"[右视图]({links['right']})"
        )
    return lines


def default_trial_links(group, trial_number, config=None):
    screenshot_dir = get_group_screenshot_dir(group, config=config)
    return {
        "top": f"{screenshot_dir}/group{group['group_no']}_trial{trial_number:02d}_top.png",
        "left": f"{screenshot_dir}/group{group['group_no']}_trial{trial_number:02d}_left.png",
        "right": f"{screenshot_dir}/group{group['group_no']}_trial{trial_number:02d}_right.png",
    }


def write_report(markdown_path, group_defs, group_results, title):
    lines = [
        f"# {title}",
        "",
        "- 物块总数：按各实验组配置动态确定",
        "",
    ]

    for group_def, group_result in zip(group_defs, group_results):
        records = group_result["records"]
        success_count = sum(1 for record in records if record["success"])
        lines.append(f"## {group_result['name']}")
        lines.append("")
        lines.append(f"- 实验次数：{get_group_trials(group_def)}")
        if records:
            if group_def["report_kind"] == "special":
                lines.append("| 次数 | 最终验收 | 原因 |")
                lines.append("| --- | --- | --- |")
                for record in records:
                    verdict = "成功" if record["success"] else "失败"
                    reason = str(record["reason"]).replace("\n", " ").strip()
                    lines.append(f"| {record['trial']} | {verdict} | {reason} |")
            else:
                lines.append("| 次数 | 大模型判定 | 原因 |")
                lines.append("| --- | --- | --- |")
                for record in records:
                    verdict = "成功" if record["success"] else "失败"
                    reason = str(record["reason"]).replace("\n", " ").strip()
                    lines.append(f"| {record['trial']} | {verdict} | {reason} |")
            lines.append("")
            lines.append(f"成功次数：{success_count}/{len(records)}")
        else:
            lines.append("当前组还没有完成的实验结果。")
        lines.append("")
        lines.append("截图链接：")
        lines.extend(render_group_links(records))
        lines.append("")

    with open(markdown_path, "w", encoding="utf-8") as markdown_file:
        markdown_file.write("\n".join(lines))


def load_existing_report(markdown_path, group_defs, group_results, config=None):
    if not os.path.exists(markdown_path):
        return

    # 使用display_name作为主键，同时保留name作为备选
    group_by_name = {}
    for index, group in enumerate(group_defs):
        # 优先使用display_name
        display_name = group.get('display_name', group['name'])
        group_by_name[display_name] = index
        # 同时也用name建立映射（兼容旧版本）
        group_by_name[group['name']] = index
    
    parsed_records = {index: {} for index in range(len(group_defs))}
    current_group_index_for_parse = None
    fallback_group_index = -1

    with open(markdown_path, "r", encoding="utf-8") as markdown_file:
        for raw_line in markdown_file:
            line = raw_line.strip()
            if line.startswith("## "):
                group_name = line[3:].strip()
                fallback_group_index += 1
                current_group_index_for_parse = group_by_name.get(group_name)
                if current_group_index_for_parse is None and fallback_group_index < len(group_defs):
                    current_group_index_for_parse = fallback_group_index
                continue

            if current_group_index_for_parse is None:
                continue

            if line.startswith("|") and not line.startswith("| ---") and "次数" not in line and "娆℃暟" not in line:
                cells = [cell.strip() for cell in line.strip("|").split("|")]
                if not cells or not cells[0].isdigit():
                    continue
                trial_number = int(cells[0])
                record = {
                    "trial": trial_number,
                    "success": False,
                    "reason": "",
                    "links": default_trial_links(group_defs[current_group_index_for_parse], trial_number, config=config),
                }
                if len(cells) >= 3:
                    record["success"] = cells[1] in SUCCESS_LABELS
                    record["reason"] = cells[2]
                parsed_records[current_group_index_for_parse][trial_number] = record
                continue

            link_match = re.match(
                r"- 第?(\d+)次.*?\[.*?\]\(([^)]+)\).*?\[.*?\]\(([^)]+)\).*?\[.*?\]\(([^)]+)\)",
                line,
            )
            if not link_match:
                link_match = re.match(
                    r"- 绗?(\d+)娆.*?\[.*?\]\(([^)]+)\).*?\[.*?\]\(([^)]+)\).*?\[.*?\]\(([^)]+)\)",
                    line,
                )
            if link_match:
                trial_number = int(link_match.group(1))
                record = parsed_records[current_group_index_for_parse].setdefault(
                    trial_number,
                    {
                        "trial": trial_number,
                        "success": False,
                        "reason": "从已有报告恢复，未解析到判定原因",
                        "links": default_trial_links(group_defs[current_group_index_for_parse], trial_number, config=config),
                    },
                )
                record["links"] = {
                    "top": link_match.group(2),
                    "left": link_match.group(3),
                    "right": link_match.group(4),
                }

    for group_index, records_by_trial in parsed_records.items():
        group_results[group_index]["records"] = [
            records_by_trial[trial_number]
            for trial_number in sorted(records_by_trial)
            if 1 <= trial_number <= get_group_trials(group_defs[group_index])
        ]


def copy_trial_screenshots(group, trial_number, image_dir, config=None):
    screenshot_dir = get_group_screenshot_dir(group, config=config)
    screenshot_targets = {
        "top": ("camera_rgb.png", f"group{group['group_no']}_trial{trial_number:02d}_top.png"),
        "left": ("camera_rgb_left.png", f"group{group['group_no']}_trial{trial_number:02d}_left.png"),
        "right": ("camera_rgb_right.png", f"group{group['group_no']}_trial{trial_number:02d}_right.png"),
    }
    copied_links = {}
    for view_name, (source_name, target_name) in screenshot_targets.items():
        source_path = os.path.join(image_dir, source_name)
        target_path = os.path.join(screenshot_dir, target_name)
        if os.path.exists(source_path):
            shutil.copy2(source_path, target_path)
        copied_links[view_name] = os.path.join(screenshot_dir, target_name).replace("\\", "/")
    return copied_links
