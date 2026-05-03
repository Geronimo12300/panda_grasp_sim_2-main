import copy
import json
import os

AUTO_RUN = True
AUTO_PLACE_CHAR = "7"
AUTO_CAPTURE_DELAY_STEPS = 240
TRIAL_TIMEOUT_STEPS = 30000
GRASP_ACTION_TIMEOUT_STEPS = 2200
PLACE_ACTION_TIMEOUT_STEPS = 6000
CONNECTION_MODE = "GUI"
SIMULATION_SLEEP_SECONDS = 0.004

RESULTS_MARKDOWN_PATH = "experiment_results.md"
SPECIAL_RESULTS_MARKDOWN_PATH = "special_experiment_results.md"
SCREENSHOT_DIR = "1"
SPECIAL_SCREENSHOT_DIR = "2"
FINAL_RENDER_DIR = "img/img_urdf"

TRIALS_PER_GROUP = 10
SPECIAL_TRIALS_PER_GROUP = 5
RESUME_FROM_GROUP_INDEX = 0
RESUME_FROM_TRIAL_INDEX = 0
RESTART_FROM_RESUME_GROUP = True

GRASP_GAP = 0.005
GRASP_DEPTH = 0.005
GRASP_WIDTH = 0.08

SCENE_SETTLE_LINEAR_VEL = 0.015
SCENE_SETTLE_ANGULAR_VEL = 0.08
SCENE_SETTLE_STEPS = 45

PAIR_SLOT_OFFSET = 0.032
LONG_BAR_PAIR_SLOT_OFFSET = 0.022
TRIANGLE_PAIR_SLOT_OFFSET = 0.014
LONG_BAR_FRONT_BACK_OFFSET = 0.020
LONG_BAR_SECOND_PLACE_HOLD_WIDTH = 0.006

BASE_RANDOM_SEED = 20260503
TRIAL_SEED_STRIDE = 1000

RUN_ROOT_DIR = "runs"
WRITE_CONFIG_SNAPSHOT = True
WRITE_TRIAL_JSONL = True
WRITE_RUN_SUMMARY = True

DATABASE_PATH = ["cube", "cylinder", "cone_top", "cuboid_bar"]

DEFAULT_CONFIG = {
    "runtime": {
        "auto_run": AUTO_RUN,
        "auto_place_char": AUTO_PLACE_CHAR,
        "auto_capture_delay_steps": AUTO_CAPTURE_DELAY_STEPS,
        "trial_timeout_steps": TRIAL_TIMEOUT_STEPS,
        "grasp_action_timeout_steps": GRASP_ACTION_TIMEOUT_STEPS,
        "place_action_timeout_steps": PLACE_ACTION_TIMEOUT_STEPS,
        "connection_mode": CONNECTION_MODE,
        "simulation_sleep_seconds": SIMULATION_SLEEP_SECONDS,
    },
    "paths": {
        "results_markdown_path": RESULTS_MARKDOWN_PATH,
        "special_results_markdown_path": SPECIAL_RESULTS_MARKDOWN_PATH,
        "screenshot_dir": SCREENSHOT_DIR,
        "special_screenshot_dir": SPECIAL_SCREENSHOT_DIR,
        "final_render_dir": FINAL_RENDER_DIR,
    },
    "resume": {
        "resume_from_group_index": RESUME_FROM_GROUP_INDEX,
        "resume_from_trial_index": RESUME_FROM_TRIAL_INDEX,
        "restart_from_resume_group": RESTART_FROM_RESUME_GROUP,
    },
    "grasp": {
        "grasp_gap": GRASP_GAP,
        "grasp_depth": GRASP_DEPTH,
        "grasp_width": GRASP_WIDTH,
    },
    "scene": {
        "settle_linear_vel": SCENE_SETTLE_LINEAR_VEL,
        "settle_angular_vel": SCENE_SETTLE_ANGULAR_VEL,
        "settle_steps": SCENE_SETTLE_STEPS,
    },
    "planner": {
        "pair_slot_offset": PAIR_SLOT_OFFSET,
        "long_bar_pair_slot_offset": LONG_BAR_PAIR_SLOT_OFFSET,
        "triangle_pair_slot_offset": TRIANGLE_PAIR_SLOT_OFFSET,
        "long_bar_front_back_offset": LONG_BAR_FRONT_BACK_OFFSET,
        "long_bar_second_place_hold_width": LONG_BAR_SECOND_PLACE_HOLD_WIDTH,
    },
    "reproducibility": {
        "base_random_seed": BASE_RANDOM_SEED,
        "trial_seed_stride": TRIAL_SEED_STRIDE,
    },
    "tracking": {
        "run_root_dir": RUN_ROOT_DIR,
        "write_config_snapshot": WRITE_CONFIG_SNAPSHOT,
        "write_trial_jsonl": WRITE_TRIAL_JSONL,
        "write_run_summary": WRITE_RUN_SUMMARY,
    },
    "database_path": list(DATABASE_PATH),
    "experiments": {
        "trials_per_group": TRIALS_PER_GROUP,
        "special_trials_per_group": SPECIAL_TRIALS_PER_GROUP,
        "categories": [
            {
                "id": "fixed_size",
                "name": "固定尺寸",
                "size_mode": "fixed",
            },
            {
                "id": "random_size",
                "name": "随机尺寸",
                "size_mode": "random",
            },
        ],
        "standard_group_templates": [
            {
                "template_id": "standard_1",
                "name": "第一组：5个正方体",
                "shapes": ["cube", "cube", "cube", "cube", "cube"],
                "report_kind": "standard",
            },
            {
                "template_id": "standard_2",
                "name": "第二组：5个圆柱体",
                "shapes": ["cylinder", "cylinder", "cylinder", "cylinder", "cylinder"],
                "report_kind": "standard",
            },
            {
                "template_id": "standard_3",
                "name": "第三组：3个正方体 + 2个圆柱体",
                "shapes": ["cube", "cube", "cube", "cylinder", "cylinder"],
                "report_kind": "standard",
            },
            {
                "template_id": "standard_4",
                "name": "第四组：2个正方体 + 2个圆柱体 + 1个三角体",
                "shapes": ["cube", "cube", "cylinder", "cylinder", "cone_top"],
                "report_kind": "standard",
            },
        ],
        "special_group_templates": [
            {
                "template_id": "special_5",
                "name": "第五组：1个正方体 + 2个细长长方体",
                "shapes": ["cube", "cuboid_bar", "cuboid_bar"],
                "report_kind": "special",
                "structure_mode": "long_bar_pair",
            },
            {
                "template_id": "special_6",
                "name": "第六组：1个正方体 + 2个三角体",
                "shapes": ["cube", "cone_top", "cone_top"],
                "report_kind": "special",
                "structure_mode": "triangle_pair_top",
            },
        ],
    },
}


def _deep_update(base, updates):
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_project_config(config_path=None):
    config = copy.deepcopy(DEFAULT_CONFIG)
    if config_path:
        with open(config_path, "r", encoding="utf-8") as config_file:
            user_config = json.load(config_file)
        _deep_update(config, user_config)
        config["_config_path"] = os.path.abspath(config_path)
    else:
        config["_config_path"] = None
    return config


def get_default_config_path(root_dir=None):
    base_dir = root_dir or os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, "config", "default.json")


def build_experiment_groups(config=None):
    config = config or DEFAULT_CONFIG
    experiment_config = config["experiments"]
    categories = copy.deepcopy(experiment_config.get("categories", []))
    standard_templates = copy.deepcopy(experiment_config["standard_group_templates"])
    special_templates = copy.deepcopy(experiment_config["special_group_templates"])
    trials_per_group = int(experiment_config["trials_per_group"])
    special_trials_per_group = int(experiment_config["special_trials_per_group"])

    standard_groups = []
    special_groups = []
    all_groups = []
    standard_result_index = 0
    special_result_index = 0
    group_no = 1

    for category in categories:
        category_id = category["id"]
        category_name = category["name"]
        size_mode = category["size_mode"]

        for template in standard_templates + special_templates:
            group = dict(template)
            group["category_id"] = category_id
            group["category_name"] = category_name
            group["size_mode"] = size_mode
            group["display_name"] = f"{category_name} - {group['name']}"
            group["trials"] = int(
                group.get(
                    "trials",
                    special_trials_per_group if group["report_kind"] == "special" else trials_per_group,
                )
            )
            group["group_no"] = group_no
            group["global_index"] = group_no - 1

            if group["report_kind"] == "special":
                group["result_index"] = special_result_index
                special_result_index += 1
                special_groups.append(group)
            else:
                group["result_index"] = standard_result_index
                standard_result_index += 1
                standard_groups.append(group)

            all_groups.append(group)
            group_no += 1

    return standard_groups, special_groups, all_groups


def build_experiment_catalog(config=None):
    _, _, all_groups = build_experiment_groups(config=config)
    catalog = {}
    for group in all_groups:
        bucket = catalog.setdefault(
            group["category_id"],
            {
                "id": group["category_id"],
                "name": group["category_name"],
                "size_mode": group["size_mode"],
                "groups": [],
            },
        )
        bucket["groups"].append(group)
    return list(catalog.values())


def get_runtime_setting(config, key):
    return config["runtime"][key]


def get_path_setting(config, key):
    return config["paths"][key]


def get_resume_setting(config, key):
    return config["resume"][key]


def get_grasp_setting(config, key):
    return config["grasp"][key]


def get_scene_setting(config, key):
    return config["scene"][key]


def get_planner_setting(config, key):
    return config["planner"][key]


def get_reproducibility_setting(config, key):
    return config["reproducibility"][key]


def get_tracking_setting(config, key):
    return config["tracking"][key]


def get_database_path(config):
    return list(config.get("database_path", DATABASE_PATH))


def get_group_trials(group):
    return int(group.get("trials", 0))


def get_group_object_count(group):
    return len(group.get("shapes", []))


def get_group_screenshot_dir(group, config=None):
    config = config or DEFAULT_CONFIG
    paths_config = config["paths"]
    return (
        paths_config["special_screenshot_dir"]
        if group["report_kind"] == "special"
        else paths_config["screenshot_dir"]
    )


def get_group_report_path(group, config=None):
    config = config or DEFAULT_CONFIG
    paths_config = config["paths"]
    return (
        paths_config["special_results_markdown_path"]
        if group["report_kind"] == "special"
        else paths_config["results_markdown_path"]
    )
