import numpy as np

from experiment_config import (
    GRASP_WIDTH,
    LONG_BAR_FRONT_BACK_OFFSET,
    LONG_BAR_PAIR_SLOT_OFFSET,
    LONG_BAR_SECOND_PLACE_HOLD_WIDTH,
    PAIR_SLOT_OFFSET,
    TRIANGLE_PAIR_SLOT_OFFSET,
)


def normalize_grasp_angle(angle):
    while angle > np.pi / 2:
        angle -= np.pi
    while angle < -np.pi / 2:
        angle += np.pi
    return angle


def clamp_value(value, lower, upper):
    return max(lower, min(upper, value))


def parse_object_index(value, object_count):
    if isinstance(value, int):
        idx = value - 1
    elif isinstance(value, str):
        digits = "".join(ch for ch in value if ch.isdigit())
        if not digits:
            return None
        idx = int(digits) - 1
    else:
        return None

    if 0 <= idx < object_count:
        return idx
    return None


def enforce_stacking_constraints(order_indices, cubes_info):
    valid_indices = [idx for idx in order_indices if 0 <= idx < len(cubes_info)]
    missing_indices = [i for i in range(len(cubes_info)) if i not in valid_indices]
    merged_order = valid_indices + missing_indices

    top_only_indices = [
        idx
        for idx in merged_order
        if cubes_info[idx].get("top_only", False) or cubes_info[idx].get("is_triangle", False)
    ]
    normal_indices = [
        idx
        for idx in merged_order
        if not (cubes_info[idx].get("top_only", False) or cubes_info[idx].get("is_triangle", False))
    ]

    normal_indices.sort(key=lambda idx: cubes_info[idx].get("volume", 0.0), reverse=True)
    top_only_indices.sort(key=lambda idx: cubes_info[idx].get("volume", 0.0), reverse=True)
    return normal_indices + top_only_indices


def get_planner_config_value(planner_config, key, default):
    if planner_config is None:
        return default
    return planner_config.get(key, default)


def build_place_pose(stack_target_xy, layer_index, slot, structure_mode="single_column", planner_config=None):
    place_x = float(stack_target_xy[0])
    place_y = float(stack_target_xy[1])
    pair_slot_offset = get_planner_config_value(planner_config, "pair_slot_offset", PAIR_SLOT_OFFSET)
    long_bar_pair_slot_offset = get_planner_config_value(
        planner_config,
        "long_bar_pair_slot_offset",
        LONG_BAR_PAIR_SLOT_OFFSET,
    )
    triangle_pair_slot_offset = get_planner_config_value(
        planner_config,
        "triangle_pair_slot_offset",
        TRIANGLE_PAIR_SLOT_OFFSET,
    )
    long_bar_front_back_offset = get_planner_config_value(
        planner_config,
        "long_bar_front_back_offset",
        LONG_BAR_FRONT_BACK_OFFSET,
    )
    if structure_mode == "long_bar_pair":
        slot_offset = long_bar_pair_slot_offset
    elif structure_mode == "triangle_pair_top":
        slot_offset = triangle_pair_slot_offset
    else:
        slot_offset = pair_slot_offset
    if slot == "left":
        place_x -= slot_offset
    elif slot == "right":
        place_x += slot_offset
    elif slot == "front":
        place_y += long_bar_front_back_offset
    elif slot == "back":
        place_y -= long_bar_front_back_offset
    return {
        "x": place_x,
        "y": place_y,
        "z": float(max(0, layer_index) * 0.04),
        "layer_index": int(layer_index),
        "slot": slot,
    }


def sort_actions_for_execution(actions):
    slot_order = {"center": 0, "front": 1, "back": 2, "left": 3, "right": 4}
    return sorted(
        actions,
        key=lambda action: (
            int(action.get("layer_index", 0)),
            slot_order.get(action.get("slot", "center"), 9),
            action["target_index"],
        ),
    )


def build_single_column_action_plan(cubes_info, stack_target_xy=(0.5, 0.0), planner_config=None):
    ordered_indices = enforce_stacking_constraints(list(range(len(cubes_info))), cubes_info)
    actions = []
    for order_rank, obj_idx in enumerate(ordered_indices):
        default_grasp_pose = cubes_info[obj_idx]["default_grasp_pose"]
        actions.append(
            {
                "target_index": obj_idx,
                "grasp_pose": dict(default_grasp_pose),
                "layer_index": order_rank,
                "slot": "center",
                "place_pose": build_place_pose(
                    stack_target_xy,
                    order_rank,
                    "center",
                    structure_mode="single_column",
                    planner_config=planner_config,
                ),
                "reason": "默认规则规划：按稳定性与形状约束单柱堆叠",
            }
        )
    return actions


def build_special_pair_action_plan(cubes_info, stack_target_xy=(0.5, 0.0), structure_mode="single_column", planner_config=None):
    if structure_mode == "long_bar_pair":
        pair_indices = [cube["index"] for cube in cubes_info if cube.get("is_long_bar")]
        pair_reason = "默认模板：两个细长长方体放在正方体堆叠点前后两侧更稳"
    else:
        pair_indices = [cube["index"] for cube in cubes_info if cube.get("is_triangle")]
        pair_reason = "默认模板：两个三角体放在正方体上层左右并排"

    normal_indices = [
        cube["index"]
        for cube in sorted(cubes_info, key=lambda item: item.get("volume", 0.0), reverse=True)
        if cube["index"] not in pair_indices
    ]
    required_normal_count = 1
    if len(normal_indices) < required_normal_count or len(pair_indices) != 2:
        return build_single_column_action_plan(
            cubes_info,
            stack_target_xy=stack_target_xy,
            planner_config=planner_config,
        )

    if structure_mode == "long_bar_pair":
        layout = [
            (pair_indices[0], 0, "front", "默认模板：细长长方体放在中心堆叠点前侧"),
            (pair_indices[1], 0, "back", "默认模板：细长长方体放在中心堆叠点后侧"),
            (normal_indices[0], 1, "center", "默认普通物块：上层中心单柱堆叠"),
        ]
    else:
        layout = [
            (normal_indices[0], 0, "center", "默认普通物块：底层中心单柱堆叠"),
            (pair_indices[0], 1, "left", pair_reason),
            (pair_indices[1], 1, "right", pair_reason),
        ]

    actions = []
    for obj_idx, layer_index, slot, reason in layout:
        place_pose = build_place_pose(
            stack_target_xy,
            layer_index,
            slot,
            structure_mode=structure_mode,
            planner_config=planner_config,
        )
        if structure_mode == "long_bar_pair" and slot == "back":
            place_pose["place_hold_width"] = get_planner_config_value(
                planner_config,
                "long_bar_second_place_hold_width",
                LONG_BAR_SECOND_PLACE_HOLD_WIDTH,
            )
        actions.append(
            {
                "target_index": obj_idx,
                "grasp_pose": dict(cubes_info[obj_idx]["default_grasp_pose"]),
                "layer_index": layer_index,
                "slot": slot,
                "place_pose": place_pose,
                "reason": reason,
            }
        )
    return sort_actions_for_execution(actions)


def build_default_action_plan(
    cubes_info,
    detected_positions,
    stack_target_xy=(0.5, 0.0),
    structure_mode="single_column",
    planner_config=None,
):
    if structure_mode in {"long_bar_pair", "triangle_pair_top"}:
        return build_special_pair_action_plan(
            cubes_info,
            stack_target_xy=stack_target_xy,
            structure_mode=structure_mode,
            planner_config=planner_config,
        )
    return build_single_column_action_plan(
        cubes_info,
        stack_target_xy=stack_target_xy,
        planner_config=planner_config,
    )


def evaluate_structure_plan(actions, cubes_info, structure_mode="single_column"):
    if structure_mode == "single_column":
        return True, "常规单柱实验"

    if structure_mode == "long_bar_pair":
        pair_actions = [action for action in actions if cubes_info[action["target_index"]].get("is_long_bar")]
        target_name = "细长长方体"
        expected_pair_slots = {"front", "back"}
        expected_slot_desc = "front/back"
    else:
        pair_actions = [action for action in actions if cubes_info[action["target_index"]].get("is_triangle")]
        target_name = "三角体"
        expected_pair_slots = {"left", "right"}
        expected_slot_desc = "left/right"

    if len(pair_actions) != 2:
        return False, f"未能识别出 2 个{target_name}动作"

    pair_layers = {int(action.get("layer_index", -1)) for action in pair_actions}
    pair_slots = {action.get("slot", "") for action in pair_actions}
    yaw_values = [float(action["grasp_pose"].get("yaw", 0.0)) for action in pair_actions]
    yaw_delta = abs(normalize_grasp_angle(yaw_values[0] - yaw_values[1]))
    if len(pair_layers) != 1:
        return False, f"{target_name}没有被规划到同一层"
    if pair_slots != expected_pair_slots:
        return False, f"{target_name}没有形成 {expected_slot_desc} 成对布局"
    if yaw_delta > 0.35:
        return False, f"{target_name}抓取朝向不够平行（差值 {yaw_delta:.3f} rad）"
    if structure_mode == "triangle_pair_top":
        highest_layer = max(int(action.get("layer_index", 0)) for action in actions)
        if next(iter(pair_layers)) != highest_layer:
            return False, "两个三角体没有被规划到最高层"
    return True, f"{target_name}满足同层并排规划要求"


def sanitize_action_plan(
    raw_actions,
    default_actions,
    object_count,
    stack_target_xy=(0.5, 0.0),
    structure_mode="single_column",
    planner_config=None,
    max_grasp_width=GRASP_WIDTH,
):
    if not raw_actions:
        return list(default_actions)

    def get_float(source, key, fallback):
        try:
            return float(source.get(key, fallback))
        except (TypeError, ValueError, AttributeError):
            return float(fallback)

    default_by_index = {action["target_index"]: action for action in default_actions}
    valid_slots = {"center", "left", "right", "front", "back"}
    planned = []
    used_indices = set()

    for raw_action in raw_actions:
        if not isinstance(raw_action, dict):
            continue
        obj_idx = parse_object_index(raw_action.get("target_object"), object_count)
        if obj_idx is None or obj_idx in used_indices or obj_idx not in default_by_index:
            continue

        default_action = default_by_index[obj_idx]
        default_grasp = default_action["grasp_pose"]
        raw_grasp = raw_action.get("grasp_pose", {}) or {}
        raw_place = raw_action.get("place_pose", {}) or {}
        try:
            layer_index = int(raw_action.get("layer_index", default_action.get("layer_index", 0)))
        except (TypeError, ValueError):
            layer_index = int(default_action.get("layer_index", 0))
        slot = str(raw_action.get("slot", default_action.get("slot", "center"))).strip().lower()
        if slot not in valid_slots:
            slot = default_action.get("slot", "center")

        grasp_pose = {
            "x": clamp_value(get_float(raw_grasp, "x", default_grasp["x"]), default_grasp["x"] - 0.05, default_grasp["x"] + 0.05),
            "y": clamp_value(get_float(raw_grasp, "y", default_grasp["y"]), default_grasp["y"] - 0.05, default_grasp["y"] + 0.05),
            "z": clamp_value(get_float(raw_grasp, "z", default_grasp["z"]), max(0.0, default_grasp["z"] - 0.02), default_grasp["z"] + 0.03),
            "yaw": float(normalize_grasp_angle(get_float(raw_grasp, "yaw", default_grasp["yaw"]))),
            "width": clamp_value(get_float(raw_grasp, "width", default_grasp["width"]), 0.02, max_grasp_width),
        }
        place_pose = build_place_pose(
            stack_target_xy,
            layer_index,
            slot,
            structure_mode=structure_mode,
            planner_config=planner_config,
        )
        place_pose["z"] = clamp_value(get_float(raw_place, "z", place_pose["z"]), 0.0, 0.25)
        if "place_hold_width" in default_action.get("place_pose", {}):
            place_pose["place_hold_width"] = float(default_action["place_pose"]["place_hold_width"])

        planned.append(
            {
                "target_index": obj_idx,
                "grasp_pose": grasp_pose,
                "layer_index": layer_index,
                "slot": slot,
                "place_pose": place_pose,
                "reason": str(raw_action.get("reason", "大模型动作规划")).strip() or "大模型动作规划",
            }
        )
        used_indices.add(obj_idx)

    for default_action in default_actions:
        if default_action["target_index"] not in used_indices:
            planned.append(default_action)

    if structure_mode in {"long_bar_pair", "triangle_pair_top"}:
        allowed_pair_indices = {
            action["target_index"]
            for action in default_actions
            if action.get("slot") in {"left", "right", "front", "back"}
        }
        normalized_planned = []
        for action in planned:
            if action["target_index"] not in allowed_pair_indices:
                default_action = default_by_index[action["target_index"]]
                action = {
                    "target_index": action["target_index"],
                    "grasp_pose": action["grasp_pose"],
                    "layer_index": default_action["layer_index"],
                    "slot": default_action["slot"],
                    "place_pose": dict(default_action["place_pose"]),
                    "reason": action["reason"],
                }
            normalized_planned.append(action)
        planned = normalized_planned

    return sort_actions_for_execution(planned)
