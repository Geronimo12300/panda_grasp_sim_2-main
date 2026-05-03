import argparse
import json
import os
import random
import time
from datetime import datetime

import numpy as np
import pybullet as p
import pybullet_data

from action_planner import (
    build_default_action_plan,
    evaluate_structure_plan,
    normalize_grasp_angle,
    sanitize_action_plan,
)
from experiment_config import (
    build_experiment_groups,
    get_database_path,
    get_default_config_path,
    get_group_object_count,
    get_group_trials,
    get_grasp_setting,
    get_path_setting,
    get_reproducibility_setting,
    get_resume_setting,
    get_runtime_setting,
    get_scene_setting,
    get_tracking_setting,
    load_project_config,
)
from experiment_report import (
    copy_trial_screenshots as report_copy_trial_screenshots,
    ensure_output_paths as report_ensure_output_paths,
    load_existing_report,
    write_report,
)
from llm_client import ask_bailian_for_pick_place_actions, ask_bailian_for_stack_success
from simEnv import SimEnv
from vision_detection import get_positions, infer_triangle_grasp_angle_from_side_views
import panda_sim_grasp as panda_sim

GRASP_MIN_WORLD_Z = 0.015
GRASP_MIN_BOTTOM_CLEARANCE = 0.007

def run(config_path=None, control_state=None):
    project_config = load_project_config(config_path)
    STANDARD_EXPERIMENT_GROUPS, SPECIAL_EXPERIMENT_GROUPS, ALL_EXPERIMENT_GROUPS = build_experiment_groups(project_config)
    database_path = get_database_path(project_config)
    auto_run = bool(get_runtime_setting(project_config, "auto_run"))
    auto_place_char = str(get_runtime_setting(project_config, "auto_place_char"))
    auto_capture_delay_steps = int(get_runtime_setting(project_config, "auto_capture_delay_steps"))
    trial_timeout_steps = int(get_runtime_setting(project_config, "trial_timeout_steps"))
    grasp_action_timeout_steps = int(get_runtime_setting(project_config, "grasp_action_timeout_steps"))
    place_action_timeout_steps = int(get_runtime_setting(project_config, "place_action_timeout_steps"))
    connection_mode_name = str(get_runtime_setting(project_config, "connection_mode")).upper()
    simulation_sleep_seconds = float(get_runtime_setting(project_config, "simulation_sleep_seconds"))
    results_markdown_path = get_path_setting(project_config, "results_markdown_path")
    special_results_markdown_path = get_path_setting(project_config, "special_results_markdown_path")
    screenshot_dir = get_path_setting(project_config, "screenshot_dir")
    special_screenshot_dir = get_path_setting(project_config, "special_screenshot_dir")
    img_path = get_path_setting(project_config, "final_render_dir")
    resume_from_group_index = int(get_resume_setting(project_config, "resume_from_group_index"))
    resume_from_trial_index = int(get_resume_setting(project_config, "resume_from_trial_index"))
    restart_from_resume_group = bool(get_resume_setting(project_config, "restart_from_resume_group"))
    grasp_gap = float(get_grasp_setting(project_config, "grasp_gap"))
    grasp_depth = float(get_grasp_setting(project_config, "grasp_depth"))
    grasp_width = float(get_grasp_setting(project_config, "grasp_width"))
    scene_settle_linear_vel = float(get_scene_setting(project_config, "settle_linear_vel"))
    scene_settle_angular_vel = float(get_scene_setting(project_config, "settle_angular_vel"))
    scene_settle_steps = int(get_scene_setting(project_config, "settle_steps"))
    base_random_seed = int(get_reproducibility_setting(project_config, "base_random_seed"))
    trial_seed_stride = int(get_reproducibility_setting(project_config, "trial_seed_stride"))
    run_root_dir = get_tracking_setting(project_config, "run_root_dir")
    write_config_snapshot = bool(get_tracking_setting(project_config, "write_config_snapshot"))
    write_trial_jsonl = bool(get_tracking_setting(project_config, "write_trial_jsonl"))
    write_run_summary = bool(get_tracking_setting(project_config, "write_run_summary"))
    planner_config = dict(project_config.get("planner", {}))
    loaded_config_path = project_config.get("_config_path") or get_default_config_path()

    print(f"当前配置文件：{loaded_config_path}")

    connection_mode = p.DIRECT if connection_mode_name == "DIRECT" else p.GUI
    p.connect(connection_mode)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    env = SimEnv(p, database_path)
    panda = panda_sim.PandaSimAuto(p, [0, -0.5, 0])

    GRASP_STATE = False
    grasp_config = {"x": 0, "y": 0, "z": 0.05, "angle": 0, "width": grasp_width}
    PLACE_STATE = False
    IN_STATE = False
    pressed_char = None
    auto_capture_pending = auto_run
    auto_capture_countdown = auto_capture_delay_steps

    ball_positions = []
    grasp_obj_index = 0
    grasp_order_indices = []
    planned_actions = []
    stack_evaluation_done = False
    auto_scene_settle_counter = 0
    trial_step_counter = 0
    grasp_action_step_counter = 0
    place_action_step_counter = 0
    current_group_index = 0
    current_trial_index = 0
    batch_finished = False
    experiment_results = [{'name': group.get('display_name', group['name']), 'records': []} for group in STANDARD_EXPERIMENT_GROUPS]
    special_experiment_results = [{'name': group.get('display_name', group['name']), 'records': []} for group in SPECIAL_EXPERIMENT_GROUPS]
    current_plan_evaluation = None
    current_trial_seed = None
    run_started_at = datetime.now().astimezone()
    run_id = f"run_{run_started_at.strftime('%Y%m%d_%H%M%S_%f')}_seed{base_random_seed}"
    run_output_dir = os.path.abspath(os.path.join(run_root_dir, run_id))
    run_config_snapshot_path = os.path.join(run_output_dir, "config_snapshot.json")
    run_metadata_path = os.path.join(run_output_dir, "run_metadata.json")
    run_summary_path = os.path.join(run_output_dir, "run_summary.json")
    trial_jsonl_path = os.path.join(run_output_dir, "trial_records.jsonl")
    latest_scene_snapshot = None
    latest_plan_snapshot = None

    def queue_auto_capture():
        nonlocal auto_capture_pending, auto_capture_countdown
        auto_capture_pending = auto_run
        auto_capture_countdown = auto_capture_delay_steps

    def make_json_safe(value):
        if isinstance(value, dict):
            return {str(key): make_json_safe(val) for key, val in value.items()}
        if isinstance(value, (list, tuple)):
            return [make_json_safe(item) for item in value]
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
        return value

    def compute_trial_seed(group_index, trial_index):
        return int(base_random_seed + group_index * trial_seed_stride + trial_index)

    def set_current_trial_seed(group_index, trial_index):
        nonlocal current_trial_seed
        current_trial_seed = compute_trial_seed(group_index, trial_index)
        random.seed(current_trial_seed)
        np.random.seed(current_trial_seed)
        return current_trial_seed

    def summarize_actions_for_log(actions):
        summary = []
        for action in actions:
            grasp_pose = action.get("grasp_pose", {})
            place_pose = action.get("place_pose", {})
            summary.append(
                {
                    "target_index": int(action.get("target_index", -1)),
                    "layer_index": int(action.get("layer_index", 0)),
                    "slot": action.get("slot", "center"),
                    "reason": action.get("reason", ""),
                    "grasp_pose": {
                        "x": float(grasp_pose.get("x", 0.0)),
                        "y": float(grasp_pose.get("y", 0.0)),
                        "z": float(grasp_pose.get("z", 0.0)),
                        "yaw": float(grasp_pose.get("yaw", 0.0)),
                        "width": float(grasp_pose.get("width", 0.0)),
                    },
                    "place_pose": {
                        "x": float(place_pose.get("x", 0.0)),
                        "y": float(place_pose.get("y", 0.0)),
                        "z": float(place_pose.get("z", 0.0)),
                        "place_hold_width": (
                            None
                            if place_pose.get("place_hold_width") is None
                            else float(place_pose.get("place_hold_width"))
                        ),
                    },
                }
            )
        return summary

    def capture_scene_snapshot():
        snapshot = []
        for idx, obj_id in enumerate(getattr(env, "urdfs_id", [])):
            position, orientation = p.getBasePositionAndOrientation(obj_id)
            linear_velocity, angular_velocity = p.getBaseVelocity(obj_id)
            snapshot.append(
                {
                    "index": idx,
                    "object_id": int(obj_id),
                    "filename": env.urdfs_filename[idx] if idx < len(env.urdfs_filename) else "",
                    "shape": env.urdfs_shapes[idx] if idx < len(env.urdfs_shapes) else "",
                    "color": env.urdfs_colors[idx] if idx < len(env.urdfs_colors) else "",
                    "scale": float(env.urdfs_scale[idx]) if idx < len(env.urdfs_scale) else 1.0,
                    "position": [float(position[0]), float(position[1]), float(position[2])],
                    "orientation_quat": [float(orientation[0]), float(orientation[1]), float(orientation[2]), float(orientation[3])],
                    "linear_velocity": [float(linear_velocity[0]), float(linear_velocity[1]), float(linear_velocity[2])],
                    "angular_velocity": [float(angular_velocity[0]), float(angular_velocity[1]), float(angular_velocity[2])],
                }
            )
        return snapshot

    def write_json_file(path, payload):
        with open(path, "w", encoding="utf-8") as file_obj:
            json.dump(make_json_safe(payload), file_obj, ensure_ascii=False, indent=2)

    def append_trial_record(record):
        if not write_trial_jsonl:
            return
        with open(trial_jsonl_path, "a", encoding="utf-8") as file_obj:
            file_obj.write(json.dumps(make_json_safe(record), ensure_ascii=False) + "\n")

    def write_run_summary_file():
        if not write_run_summary:
            return
        summary_payload = {
            "run_id": run_id,
            "config_path": loaded_config_path,
            "base_random_seed": base_random_seed,
            "trial_seed_stride": trial_seed_stride,
            "started_at": run_started_at.isoformat(),
            "current_group_index": current_group_index,
            "current_trial_index": current_trial_index,
            "batch_finished": batch_finished,
            "standard_results": experiment_results,
            "special_results": special_experiment_results,
        }
        write_json_file(run_summary_path, summary_payload)

    def initialize_run_tracking():
        os.makedirs(run_output_dir, exist_ok=True)
        metadata = {
            "run_id": run_id,
            "started_at": run_started_at.isoformat(),
            "config_path": loaded_config_path,
            "base_random_seed": base_random_seed,
            "trial_seed_stride": trial_seed_stride,
            "run_output_dir": run_output_dir,
        }
        write_json_file(run_metadata_path, metadata)
        if write_config_snapshot:
            write_json_file(run_config_snapshot_path, project_config)
        if write_trial_jsonl and not os.path.exists(trial_jsonl_path):
            with open(trial_jsonl_path, "w", encoding="utf-8") as _:
                pass
        print(f"实验运行目录：{run_output_dir}")

    def ensure_output_paths():
        report_ensure_output_paths(screenshot_dir, special_screenshot_dir, img_path)

    def get_group_result_entry(group):
        if group['report_kind'] == 'special':
            return special_experiment_results[group['result_index']]
        return experiment_results[group['result_index']]

    def write_markdown_report():
        write_report(results_markdown_path, STANDARD_EXPERIMENT_GROUPS, experiment_results, "自动化堆叠实验结果")

    def write_special_markdown_report():
        write_report(
            special_results_markdown_path,
            SPECIAL_EXPERIMENT_GROUPS,
            special_experiment_results,
            "非常规结构实验结果",
        )


    def load_existing_markdown_report():
        load_existing_report(
            results_markdown_path,
            STANDARD_EXPERIMENT_GROUPS,
            experiment_results,
            config=project_config,
        )

    def load_existing_special_report():
        load_existing_report(
            special_results_markdown_path,
            SPECIAL_EXPERIMENT_GROUPS,
            special_experiment_results,
            config=project_config,
        )

    def prepare_resume_state():
        nonlocal current_group_index, current_trial_index, batch_finished

        resume_group_index = max(0, min(resume_from_group_index, len(ALL_EXPERIMENT_GROUPS) - 1))
        resume_trial_index = max(0, min(resume_from_trial_index, get_group_trials(ALL_EXPERIMENT_GROUPS[resume_group_index]) - 1))
        if restart_from_resume_group:
            resume_group = ALL_EXPERIMENT_GROUPS[resume_group_index]
            get_group_result_entry(resume_group)['records'] = [
                record
                for record in get_group_result_entry(resume_group)['records']
                if record['trial'] <= resume_trial_index
            ]
            for group_index in range(resume_group_index + 1, len(ALL_EXPERIMENT_GROUPS)):
                get_group_result_entry(ALL_EXPERIMENT_GROUPS[group_index])['records'] = []
            current_group_index = resume_group_index
            current_trial_index = resume_trial_index
            print(
                f"已保留前面记录，并保留当前组前 {resume_trial_index} 次记录，将从 "
                f"{ALL_EXPERIMENT_GROUPS[current_group_index].get('display_name', ALL_EXPERIMENT_GROUPS[current_group_index]['name'])} 第 {current_trial_index + 1} 次重新开始。"
            )
            return

        for group_index in range(resume_group_index, len(ALL_EXPERIMENT_GROUPS)):
            group = ALL_EXPERIMENT_GROUPS[group_index]
            completed_trials = {record['trial'] for record in get_group_result_entry(group)['records']}
            for trial_index in range(get_group_trials(group)):
                if trial_index + 1 not in completed_trials:
                    current_group_index = group_index
                    current_trial_index = trial_index
                    print(
                        f"已加载已有记录，将从 {ALL_EXPERIMENT_GROUPS[current_group_index].get('display_name', ALL_EXPERIMENT_GROUPS[current_group_index]['name'])} "
                        f"第 {current_trial_index + 1} 次继续。"
                    )
                    return

        batch_finished = True
        print('所有实验记录已经全部完成。')

    def copy_trial_screenshots(group, trial_number):
        return report_copy_trial_screenshots(group, trial_number, img_path, config=project_config)

    def reset_robot_and_runtime():
        nonlocal GRASP_STATE, PLACE_STATE, IN_STATE, pressed_char
        nonlocal ball_positions, grasp_obj_index, grasp_order_indices, planned_actions
        nonlocal stack_evaluation_done, auto_scene_settle_counter, trial_step_counter
        nonlocal grasp_action_step_counter, place_action_step_counter, current_plan_evaluation
        nonlocal latest_scene_snapshot, latest_plan_snapshot

        panda.reset_to_initial_pose(settle_steps=40)
        GRASP_STATE = False
        PLACE_STATE = False
        IN_STATE = False
        pressed_char = None
        ball_positions = []
        grasp_obj_index = 0
        grasp_order_indices = []
        planned_actions = []
        stack_evaluation_done = False
        current_plan_evaluation = None
        latest_plan_snapshot = None
        auto_scene_settle_counter = 0
        trial_step_counter = 0
        grasp_action_step_counter = 0
        place_action_step_counter = 0
        latest_scene_snapshot = None
        queue_auto_capture()

    def clear_world_before_trial():
        panda.reset_to_initial_pose(settle_steps=120)
        if getattr(env, 'urdfs_id', None):
            env.removeObjsInURDF()
        for _ in range(60):
            p.stepSimulation()
        panda.reset_to_initial_pose(settle_steps=20)

    def start_current_trial():
        nonlocal latest_scene_snapshot
        group = ALL_EXPERIMENT_GROUPS[current_group_index]
        group_obj_count = get_group_object_count(group)
        seed = set_current_trial_seed(current_group_index, current_trial_index)
        print('\n' + '=' * 60)
        print(f"开始实验：{group.get('display_name', group['name'])}，第 {current_trial_index + 1}/{get_group_trials(group)} 次")
        print(f"物块组合：{group['shapes']}")
        print(f"尺寸模式：{group.get('category_name', '未分类')} / {group.get('size_mode', 'random')}")
        print(f"本次实验随机种子：{seed}")
        print('=' * 60)
        clear_world_before_trial()
        env.loadObjsInURDF(0, group_obj_count, shape_sequence=group['shapes'], scale_mode=group.get('size_mode', 'random'))
        reset_robot_and_runtime()
        latest_scene_snapshot = capture_scene_snapshot()

    def build_default_grasp_pose(obj_idx, detected_pos):
        detected_x, detected_y, detected_z, detected_angle, detected_width = detected_pos
        current_group = ALL_EXPERIMENT_GROUPS[current_group_index]
        small_cube_edge_threshold = 0.034
        small_cylinder_diameter_threshold = 0.040
        pose = {
            'x': float(detected_x),
            'y': float(detected_y),
            'z': float(detected_z),
            'yaw': float(detected_angle),
            'width': float(detected_width),
        }

        if obj_idx >= len(env.urdfs_id):
            return pose

        target_obj_id = env.urdfs_id[obj_idx]
        aabb_min, aabb_max = p.getAABB(target_obj_id)
        size_x = aabb_max[0] - aabb_min[0]
        size_y = aabb_max[1] - aabb_min[1]
        size_z = aabb_max[2] - aabb_min[2]
        pose['x'] = float((aabb_min[0] + aabb_max[0]) / 2.0)
        pose['y'] = float((aabb_min[1] + aabb_max[1]) / 2.0)
        pose['z'] = float((aabb_min[2] + aabb_max[2]) / 2.0)

        _, obj_orn = p.getBasePositionAndOrientation(target_obj_id)
        obj_yaw = p.getEulerFromQuaternion(obj_orn)[2]
        if obj_yaw > np.pi / 2:
            obj_yaw -= np.pi
        elif obj_yaw < -np.pi / 2:
            obj_yaw += np.pi
        pose['yaw'] = float(obj_yaw)

        target_filename = os.path.basename(env.urdfs_filename[obj_idx]) if obj_idx < len(env.urdfs_filename) else ''
        if target_filename.startswith('cube'):
            cube_edge = size_z
            if cube_edge <= small_cube_edge_threshold:
                pose["width"] = float(np.clip(cube_edge + 1.2 * grasp_gap, 0.020, 0.034))
                pose['z'] = float(aabb_min[2] + 0.46 * size_z)
            else:
                pose["width"] = float(np.clip(cube_edge + 2 * grasp_gap, 0.025, grasp_width))
                pose['z'] = float(aabb_min[2] + 0.38 * size_z)
        elif target_filename.startswith('cuboid_bar'):
            grasp_span = max(size_x, size_y)
            pose["width"] = float(np.clip(grasp_span + 2 * grasp_gap, 0.022, 0.055))
            pose['z'] = float(aabb_min[2] + 0.45 * size_z)
        elif target_filename.startswith('cylinder'):
            cylinder_diameter = max(size_x, size_y)
            if cylinder_diameter <= small_cylinder_diameter_threshold:
                pose["width"] = float(np.clip(cylinder_diameter + 0.012, 0.028, 0.038))
                pose['z'] = float(aabb_min[2] + 0.48 * size_z)
            else:
                pose["width"] = float(np.clip(cylinder_diameter + 0.025, 0.045, grasp_width))
                pose['z'] = float(aabb_min[2] + 0.42 * size_z)
            pose['yaw'] = 0.0
        elif target_filename.startswith('cone_top'):
            flat_face_span = min(size_x, size_y)
            pose["width"] = float(np.clip(flat_face_span + 2 * grasp_gap, 0.02, grasp_width))
            target_color = env.urdfs_colors[obj_idx] if obj_idx < len(env.urdfs_colors) else None
            adjusted_angle, _ = infer_triangle_grasp_angle_from_side_views(img_path, target_color, pose['yaw'])
            fallback_angle = normalize_grasp_angle(float(adjusted_angle) + np.pi / 2)
            if (
                current_group.get("size_mode") == "fixed"
                and current_group.get("template_id") == "standard_4"
            ):
                pose['yaw'] = float(fallback_angle)
                pose['_triangle_backup_yaw'] = float(adjusted_angle)
            else:
                pose['yaw'] = float(adjusted_angle)
                pose['_triangle_backup_yaw'] = float(fallback_angle)

        min_pose_z = max(
            float(aabb_min[2] + grasp_depth + GRASP_MIN_BOTTOM_CLEARANCE),
            float(GRASP_MIN_WORLD_Z + grasp_depth),
        )
        pose['z'] = float(max(pose['z'], min_pose_z))
        return pose

    def advance_to_next_trial():
        nonlocal current_group_index, current_trial_index, batch_finished
        current_group = ALL_EXPERIMENT_GROUPS[current_group_index]

        if current_trial_index + 1 < get_group_trials(current_group):
            current_trial_index += 1
            start_current_trial()
            return

        if current_group_index + 1 < len(ALL_EXPERIMENT_GROUPS):
            current_group_index += 1
            current_trial_index = 0
            start_current_trial()
            return

        batch_finished = True
        print(f"结构化运行记录目录：{run_output_dir}")
        write_run_summary_file()
        print('\n所有自动化实验均已完成。')
        print(f"标准实验结果已写入：{os.path.abspath(results_markdown_path)}")
        print(f"标准实验截图目录：{os.path.abspath(screenshot_dir)}")
        print(f"非常规实验结果已写入：{os.path.abspath(special_results_markdown_path)}")
        print(f"非常规实验截图目录：{os.path.abspath(special_screenshot_dir)}")

    def finalize_current_trial(forced_failure_reason=None):
        nonlocal stack_evaluation_done, current_plan_evaluation, latest_scene_snapshot, latest_plan_snapshot

        if stack_evaluation_done:
            return

        stack_evaluation_done = True
        current_group = ALL_EXPERIMENT_GROUPS[current_group_index]
        group_obj_count = get_group_object_count(current_group)
        env.renderURDFImage(save_path=img_path)
        evaluation_images = [
            os.path.join(img_path, 'camera_rgb.png'),
            os.path.join(img_path, 'camera_rgb_left.png'),
            os.path.join(img_path, 'camera_rgb_right.png')
        ]
        model_success, model_reason = ask_bailian_for_stack_success(
            image_paths=evaluation_images,
            expected_count=group_obj_count
        )
        success = model_success if forced_failure_reason is None else False
        reason = model_reason if forced_failure_reason is None else f"{forced_failure_reason}; 大模型判断：{model_reason}"
        links = copy_trial_screenshots(current_group, current_trial_index + 1)
        if current_plan_evaluation is None:
            if current_group['report_kind'] == 'special':
                plan_evaluation = {'success': False, 'reason': '本轮未完成非常规结构规划'}
            else:
                plan_evaluation = {'success': True, 'reason': '常规实验未单独检查规划结构'}
        else:
            plan_evaluation = current_plan_evaluation
        record = {
            'trial': current_trial_index + 1,
            'success': success,
            'reason': reason,
            'links': links,
        }
        if current_group['report_kind'] == 'special':
            record['planning_success'] = bool(plan_evaluation.get('success'))
            record['planning_reason'] = str(plan_evaluation.get('reason', '')).strip()
        get_group_result_entry(current_group)['records'].append(record)
        append_trial_record(
            {
                'run_id': run_id,
                'group_index': current_group_index,
                'group_name': current_group.get('display_name', current_group['name']),
                'category_name': current_group.get('category_name'),
                'size_mode': current_group.get('size_mode'),
                'report_kind': current_group['report_kind'],
                'trial_index': current_trial_index,
                'trial_number': current_trial_index + 1,
                'seed': current_trial_seed,
                'shapes': list(current_group.get('shapes', [])),
                'structure_mode': current_group.get('structure_mode', 'single_column'),
                'success': bool(success),
                'reason': reason,
                'planning_success': bool(plan_evaluation.get('success')),
                'planning_reason': str(plan_evaluation.get('reason', '')).strip(),
                'links': links,
                'scene_snapshot': latest_scene_snapshot,
                'plan_snapshot': latest_plan_snapshot,
            }
        )
        write_markdown_report()
        write_special_markdown_report()
        write_run_summary_file()

        print('\n' + '=' * 50)
        print('【堆叠验收结果】')
        print(f"实验组别: {current_group.get('display_name', current_group['name'])}")
        print(f"实验次数: 第 {current_trial_index + 1} 次")
        if current_group['report_kind'] == 'special':
            print(f"规划判定: {'符合' if plan_evaluation.get('success') else '不符合'}")
            print(f"规划摘要: {plan_evaluation.get('reason', '')}")
        print(f"是否成功: {'成功' if success else '失败'}")
        print(f"原因: {reason}")
        print('=' * 50 + '\n')
        current_plan_evaluation = None
        latest_plan_snapshot = None

        advance_to_next_trial()

    def scene_is_settled():
        if not env.urdfs_id:
            return False

        for obj_id in env.urdfs_id:
            linear_vel, angular_vel = p.getBaseVelocity(obj_id)
            if np.linalg.norm(linear_vel) > scene_settle_linear_vel:
                return False
            if np.linalg.norm(angular_vel) > scene_settle_angular_vel:
                return False
        return True

    def plan_scene():
        nonlocal ball_positions, grasp_order_indices, grasp_obj_index, planned_actions, stack_evaluation_done
        nonlocal current_plan_evaluation, latest_scene_snapshot, latest_plan_snapshot

        env.renderURDFImage(save_path=img_path)
        detected_positions = get_positions(img_path, env.urdfs_id, max_grasp_width=grasp_width)
        print(f"Mask 检测到的物块数量: {len(detected_positions)}/{len(env.urdfs_id)}")

        ball_positions = []
        grasp_order_indices = []
        planned_actions = []
        grasp_obj_index = 0
        stack_evaluation_done = False
        current_plan_evaluation = None
        latest_plan_snapshot = None
        latest_scene_snapshot = capture_scene_snapshot()
        if len(detected_positions) != len(env.urdfs_id):
            missing = [i + 1 for i in range(len(env.urdfs_id)) if i not in detected_positions]
            print(f"Mask 未检测到全部物块，缺失物块编号: {missing}，等待重新渲染。")
            return False

        cubes_info = []

        print('\n' + '=' * 50)
        print('【仿真环境中的物块信息】')

        for i in range(len(env.urdfs_id)):
            scale = env.urdfs_scale[i] if i < len(env.urdfs_scale) else 1.0
            obj_pos, _ = p.getBasePositionAndOrientation(env.urdfs_id[i])
            color = env.urdfs_colors[i] if i < len(env.urdfs_colors) else f'物块{i+1}'
            shape = env.urdfs_shapes[i] if hasattr(env, 'urdfs_shapes') and i < len(env.urdfs_shapes) else '正方体'
            source_filename = os.path.basename(env.urdfs_filename[i]) if i < len(env.urdfs_filename) else ''
            is_triangle = source_filename.startswith('cone_top')
            is_long_bar = source_filename.startswith('cuboid_bar') or shape == '细长长方体'
            top_only = is_triangle or shape == '三角体'
            aabb = p.getAABB(env.urdfs_id[i])
            size_x = aabb[1][0] - aabb[0][0]
            size_y = aabb[1][1] - aabb[0][1]
            size_z = aabb[1][2] - aabb[0][2]
            footprint_x = min(size_x, size_y)
            footprint_y = max(size_x, size_y)
            footprint_area = footprint_x * footprint_y
            slenderness_ratio = size_z / max(footprint_x, 1e-6)

            shape_params = {}
            if is_long_bar:
                volume = size_x * size_y * size_z
                shape_params.update({
                    'length': footprint_y,
                    'width_body': footprint_x,
                    'height': size_z,
                    'volume': volume,
                })
                shape_summary = (
                    f"底面={footprint_x:.4f}m x {footprint_y:.4f}m, "
                    f"高度={size_z:.4f}m, 体积={volume:.8f}m^3"
                )
            elif shape == '圆柱体':
                diameter = max(size_x, size_y)
                height = size_z
                volume = np.pi * (diameter / 2.0) ** 2 * height
                shape_params.update({'diameter': diameter, 'height': height, 'volume': volume})
                shape_summary = f"直径={diameter:.4f}m, 高度={height:.4f}m, 体积={volume:.8f}m^3"
            elif shape == '三角体':
                base_diameter = max(size_x, size_y)
                height = size_z
                volume = (np.pi * (base_diameter / 2.0) ** 2 * height) / 3.0
                shape_params.update({'base_diameter': base_diameter, 'height': height, 'volume': volume})
                shape_summary = f"底面直径={base_diameter:.4f}m, 高度={height:.4f}m, 体积={volume:.8f}m^3"
            else:
                edge_length = max(size_x, size_y, size_z)
                volume = edge_length ** 3
                shape_params.update({'edge_length': edge_length, 'volume': volume})
                shape_summary = f"边长={edge_length:.4f}m, 体积={volume:.8f}m^3"

            print(
                f"  物块{i+1}: 形状={shape}, 颜色={color}, 缩放比例={scale:.2f}, "
                f"{shape_summary}, 位置=({obj_pos[0]:.3f}, {obj_pos[1]:.3f})"
            )

            mask_pos = detected_positions[i]
            cube_info = {
                'index': i,
                'scale': scale,
                'color': color,
                'shape': shape,
                'top_only': top_only,
                'is_triangle': is_triangle,
                'is_long_bar': is_long_bar,
                'source_filename': source_filename,
                'position': [obj_pos[0], obj_pos[1]],
                'footprint_x': footprint_x,
                'footprint_y': footprint_y,
                'footprint_area': footprint_area,
                'height': size_z,
                'slenderness_ratio': slenderness_ratio,
            }
            cube_info['default_grasp_pose'] = build_default_grasp_pose(i, mask_pos)
            cube_info.update(shape_params)
            cubes_info.append(cube_info)
            print(f"    -> Mask 抓取候选 ({mask_pos[0]:.3f}, {mask_pos[1]:.3f})")

        print('=' * 50 + '\n')
        print('正在询问阿里云百炼 / Qwen-VL 直接生成抓取放置动作...')

        if not cubes_info:
            return False

        current_group = ALL_EXPERIMENT_GROUPS[current_group_index]
        structure_mode = (
            current_group.get('structure_mode', 'single_column')
            if current_group.get('report_kind') == 'special'
            else 'single_column'
        )
        stack_target_xy = (0.5, 0.0)
        scene_image_paths = [
            os.path.join(img_path, 'camera_rgb.png'),
            os.path.join(img_path, 'camera_rgb_left.png'),
            os.path.join(img_path, 'camera_rgb_right.png')
        ]
        default_actions = build_default_action_plan(
            cubes_info,
            detected_positions,
            stack_target_xy=stack_target_xy,
            structure_mode=structure_mode,
            planner_config=planner_config,
        )
        forced_default_slot_indices = set()
        if structure_mode == 'single_column':
            triangle_indices = [cube['index'] for cube in cubes_info if cube.get('is_triangle')]
            if len(triangle_indices) == 1:
                forced_default_slot_indices.add(triangle_indices[0])
        raw_actions = ask_bailian_for_pick_place_actions(
            cubes_info,
            image_paths=scene_image_paths,
            stack_target={'x': stack_target_xy[0], 'y': stack_target_xy[1], 'z': 0.0},
            structure_mode=structure_mode,
        ) or []
        candidate_actions = sanitize_action_plan(
            raw_actions,
            default_actions,
            len(cubes_info),
            stack_target_xy=stack_target_xy,
            structure_mode=structure_mode,
            planner_config=planner_config,
            max_grasp_width=grasp_width,
            forced_default_slot_indices=forced_default_slot_indices,
        )
        planning_success, planning_reason = evaluate_structure_plan(candidate_actions, cubes_info, structure_mode=structure_mode)
        if structure_mode != 'single_column' and not raw_actions:
            planning_success = False
            planning_reason = '大模型未返回有效非常规结构动作，已回退默认模板'
        current_plan_evaluation = {
            'success': planning_success,
            'reason': planning_reason,
        }
        planned_actions = candidate_actions if planning_success or structure_mode == 'single_column' else list(default_actions)
        latest_plan_snapshot = {
            'structure_mode': structure_mode,
            'stack_target_xy': [float(stack_target_xy[0]), float(stack_target_xy[1])],
            'planning_success': bool(planning_success),
            'planning_reason': planning_reason,
            'raw_action_count': len(raw_actions),
            'default_actions': summarize_actions_for_log(default_actions),
            'candidate_actions': summarize_actions_for_log(candidate_actions),
            'final_actions': summarize_actions_for_log(planned_actions),
            'cubes_info': make_json_safe(cubes_info),
            'detected_positions': make_json_safe(detected_positions),
        }
        grasp_order_indices = [action['target_index'] for action in planned_actions]
        ball_positions = [
            (
                action['grasp_pose']['x'],
                action['grasp_pose']['y'],
                action['grasp_pose']['z'],
                action['grasp_pose']['yaw'],
                action['grasp_pose']['width'],
            )
            for action in planned_actions
        ]

        print("\n" + "=" * 50)
        print("【最终动作计划】")
        for action_idx, action in enumerate(planned_actions, 1):
            grasp_pose = action['grasp_pose']
            place_pose = action['place_pose']
            print(
                f"  Action {action_idx}: 物块{action['target_index'] + 1}, "
                f"grasp=({grasp_pose['x']:.3f}, {grasp_pose['y']:.3f}, {grasp_pose['z']:.3f}, "
                f"yaw={grasp_pose['yaw']:.3f}, width={grasp_pose['width']:.3f}), "
                f"layer={action.get('layer_index', 0)}, slot={action.get('slot', 'center')}, "
                f"place=({place_pose['x']:.3f}, {place_pose['y']:.3f}, {place_pose['z']:.3f}), "
                f"reason={action['reason']}"
            )
        print("=" * 50 + "\n")
        return len(planned_actions) > 0

    def begin_next_grasp():
        nonlocal GRASP_STATE, grasp_action_step_counter

        if not planned_actions:
            print('未生成动作计划，请先完成场景规划。')
            return False

        if grasp_obj_index >= len(planned_actions):
            return False

        current_action = planned_actions[grasp_obj_index]
        grasp_pose = current_action['grasp_pose']
        grasp_config['x'] = grasp_pose['x']
        grasp_config['y'] = grasp_pose['y']
        grasp_config['z'] = grasp_pose['z']
        grasp_config['angle'] = grasp_pose['yaw']
        grasp_config['width'] = grasp_pose['width']

        target_obj_id = None
        obj_idx = current_action['target_index']
        if obj_idx < len(env.urdfs_id):
            target_obj_id = env.urdfs_id[obj_idx]

        if target_obj_id is not None:
            print(
                f"执行动作: 抓取物块{obj_idx + 1}, "
                f"({grasp_config['x']:.4f}, {grasp_config['y']:.4f}, {grasp_config['z']:.4f}), "
                f"抓取角度={grasp_config['angle']:.4f} rad, "
                f"夹爪宽度={grasp_config['width']:.4f}m, "
                f"层={current_action.get('layer_index', 0)}, 槽位={current_action.get('slot', 'center')}, "
                f"放置点=({current_action['place_pose']['x']:.4f}, {current_action['place_pose']['y']:.4f}, {current_action['place_pose']['z']:.4f})"
            )

        GRASP_STATE = True
        grasp_action_step_counter = 0
        return True

    def abort_current_grasp(reason):
        nonlocal GRASP_STATE, ball_positions, grasp_order_indices, planned_actions, auto_scene_settle_counter
        nonlocal grasp_action_step_counter

        print(reason)
        panda.force_release_held_object()
        panda.reset()
        GRASP_STATE = False
        grasp_action_step_counter = 0
        if auto_run:
            ball_positions = []
            grasp_order_indices = []
            planned_actions = []
            auto_scene_settle_counter = 0
            queue_auto_capture()

    def abort_current_place(reason):
        nonlocal PLACE_STATE, place_action_step_counter

        print(reason)
        panda.force_release_held_object()
        for _ in range(80):
            p.stepSimulation()
        PLACE_STATE = False
        place_action_step_counter = 0
        if auto_run:
            finalize_current_trial(forced_failure_reason=reason)

    ensure_output_paths()
    initialize_run_tracking()
    load_existing_markdown_report()
    load_existing_special_report()
    prepare_resume_state()
    write_markdown_report()
    write_special_markdown_report()
    write_run_summary_file()
    if not batch_finished:
        start_current_trial()

    if auto_run:
        print('Auto run enabled: experiment scheduler is active.')

    while True:
        if batch_finished:
            break

        if control_state is not None:
            reset_requested, requested_group_index, requested_trial_index = control_state.consume_reset_request()
            if reset_requested:
                if requested_group_index is not None and 0 <= requested_group_index < len(ALL_EXPERIMENT_GROUPS):
                    current_group_index = requested_group_index
                    max_trials = get_group_trials(ALL_EXPERIMENT_GROUPS[current_group_index])
                    requested_trial_index = 0 if requested_trial_index is None else requested_trial_index
                    current_trial_index = max(0, min(requested_trial_index, max_trials - 1))
                    batch_finished = False
                    print(
                        f"收到控制面板重置请求，切换到 "
                        f"{ALL_EXPERIMENT_GROUPS[current_group_index].get('display_name', ALL_EXPERIMENT_GROUPS[current_group_index]['name'])} 第 {current_trial_index + 1} 次。"
                    )
                else:
                    print("收到控制面板重置请求，正在重置当前实验。")
                start_current_trial()
                continue
            if control_state.is_paused():
                time.sleep(0.1)
                continue

        p.stepSimulation()
        if simulation_sleep_seconds > 0:
            time.sleep(simulation_sleep_seconds)
        trial_step_counter += 1

        keys = p.getKeyboardEvents()

        if scene_is_settled():
            auto_scene_settle_counter += 1
        else:
            auto_scene_settle_counter = 0

        if auto_run and auto_capture_pending and not GRASP_STATE and not PLACE_STATE and not IN_STATE:
            auto_capture_countdown -= 1
            if auto_capture_countdown <= 0 and auto_scene_settle_counter >= scene_settle_steps:
                if plan_scene():
                    auto_capture_pending = False
                    auto_scene_settle_counter = 0
                else:
                    auto_capture_countdown = auto_capture_delay_steps

        if not auto_run and ord('1') in keys and keys[ord('1')] & p.KEY_WAS_TRIGGERED:
            plan_scene()

        if auto_run and not auto_capture_pending and not GRASP_STATE and not PLACE_STATE and not IN_STATE:
            if planned_actions and grasp_obj_index < len(planned_actions) and auto_scene_settle_counter >= scene_settle_steps:
                begin_next_grasp()
                auto_scene_settle_counter = 0

        if not auto_run and ord('2') in keys and keys[ord('2')] & p.KEY_WAS_TRIGGERED:
            begin_next_grasp()

        if GRASP_STATE:
            target_z = max(float(grasp_config["z"] - grasp_depth), GRASP_MIN_WORLD_Z)
            target_position = [grasp_config["x"], grasp_config["y"], target_z]
            current_held_obj_id = None
            if grasp_obj_index < len(grasp_order_indices):
                current_obj_idx = grasp_order_indices[grasp_obj_index]
                if current_obj_idx < len(env.urdfs_id):
                    current_held_obj_id = env.urdfs_id[current_obj_idx]

            if panda.grasp_step(target_position, grasp_config['angle'], grasp_config['width'], current_held_obj_id):
                GRASP_STATE = False
                if panda.held_object_id is None:
                    current_action = planned_actions[grasp_obj_index] if grasp_obj_index < len(planned_actions) else None
                    current_obj_idx = grasp_order_indices[grasp_obj_index] if grasp_obj_index < len(grasp_order_indices) else None
                    source_filename = ""
                    if current_obj_idx is not None and current_obj_idx < len(env.urdfs_filename):
                        source_filename = os.path.basename(env.urdfs_filename[current_obj_idx])
                    if (
                        current_action is not None
                        and source_filename.startswith("cone_top")
                        and not current_action.get("_triangle_retry_used", False)
                    ):
                        current_action["_triangle_retry_used"] = True
                        backup_yaw = current_action["grasp_pose"].get("_triangle_backup_yaw")
                        if backup_yaw is None:
                            backup_yaw = normalize_grasp_angle(
                                float(current_action["grasp_pose"].get("yaw", 0.0)) + np.pi / 2
                            )
                        current_action["grasp_pose"]["yaw"] = float(backup_yaw)
                        panda.reset()
                        print("三角体首次抓取失败，切换备用抓取方向后重试当前物块。")
                        if auto_run:
                            auto_scene_settle_counter = scene_settle_steps
                        else:
                            begin_next_grasp()
                        continue
                    panda.reset()
                    print('抓取失败：未抓稳目标物块，重新规划当前场景。')
                    if auto_run:
                        ball_positions = []
                        grasp_order_indices = []
                        auto_scene_settle_counter = 0
                        queue_auto_capture()
                    else:
                        IN_STATE = False
                else:
                    if auto_run:
                        pressed_char = auto_place_char
                        current_group = ALL_EXPERIMENT_GROUPS[current_group_index]
                        place_pose = dict(planned_actions[grasp_obj_index]['place_pose'])
                        if current_group.get('report_kind') != 'special':
                            place_pose.setdefault('approach_clearance', 0.06)
                            place_pose.setdefault('retreat_lift_delta', 0.05)
                            place_pose.setdefault('retreat_height', 0.0)
                        panda.set_place_target_override(place_pose)
                        panda.start_place()
                        place_action_step_counter = 0
                        PLACE_STATE = True
                    else:
                        IN_STATE = True
                    print('抓取完成。')

        if IN_STATE:
            for char in ['7', '8', '9', '0']:
                key_code = ord(char)
                if key_code in keys and keys[key_code] & p.KEY_WAS_TRIGGERED:
                    pressed_char = char
                    panda.set_place_target_override(None)
                    panda.start_place()
                    place_action_step_counter = 0
                    IN_STATE = False
                    PLACE_STATE = True

        if PLACE_STATE:
            if grasp_obj_index < len(grasp_order_indices):
                obj_idx = grasp_order_indices[grasp_obj_index]
                held_obj_id = env.urdfs_id[obj_idx] if obj_idx < len(env.urdfs_id) else None
            else:
                held_obj_id = None
            if panda.place_step(pressed_char, held_obj_id):
                PLACE_STATE = False
                place_action_step_counter = 0
                grasp_obj_index += 1
                print('放置完成。')
                if auto_run and grasp_obj_index >= len(grasp_order_indices):
                    print('Auto run finished: all objects processed.')
                if grasp_obj_index >= len(grasp_order_indices):
                    finalize_current_trial()

        if GRASP_STATE:
            grasp_action_step_counter += 1
            if grasp_action_step_counter > grasp_action_timeout_steps:
                abort_current_grasp('抓取动作超时：已强制释放并重新规划当前场景。')

        if PLACE_STATE:
            place_action_step_counter += 1
            if place_action_step_counter > place_action_timeout_steps:
                abort_current_place('放置动作超时：已强制松爪，当前小次实验按失败记录并进入下一次。')

        if auto_run and trial_step_counter > trial_timeout_steps:
            print('当前实验超过时长上限，按失败记录并自动进入下一次。')
            finalize_current_trial(forced_failure_reason='实验超时，未在限定步数内完成堆叠')

        if ord('3') in keys and keys[ord('3')] & p.KEY_WAS_TRIGGERED:
            start_current_trial()
            print('环境已重置。')

    p.disconnect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Panda 抓取堆叠仿真入口")
    parser.add_argument(
        "--config",
        default=get_default_config_path(),
        help="JSON 配置文件路径",
    )
    args = parser.parse_args()
    run(config_path=args.config)
