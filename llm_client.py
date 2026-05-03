import base64
import json
import os

import requests

from action_planner import enforce_stacking_constraints

BAILIAN_API_KEY = "sk-4ea064d0eb6b4c39b6ae8479e8975443"
BAILIAN_API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"


def encode_image_to_data_url(image_path):
    if not image_path or not os.path.exists(image_path):
        return None

    suffix = os.path.splitext(image_path)[1].lower()
    mime_type = "image/png" if suffix == ".png" else "image/jpeg"
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def _extract_json_text(content):
    json_match = content
    if "```json" in content:
        json_match = content.split("```json")[1].split("```")[0]
    elif "```" in content:
        json_match = content.split("```")[1].split("```")[0]
    json_match = json_match.strip()
    first_brace = json_match.find("{")
    last_brace = json_match.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace >= first_brace:
        json_match = json_match[first_brace:last_brace + 1]
    return json_match.strip()


def _load_json_lenient(content):
    json_text = _extract_json_text(content)
    try:
        return json.loads(json_text)
    except json.JSONDecodeError:
        open_braces = json_text.count("{")
        close_braces = json_text.count("}")
        if open_braces > close_braces:
            repaired = json_text + ("}" * (open_braces - close_braces))
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                pass
        salvaged_actions = _salvage_partial_actions(json_text)
        if salvaged_actions:
            return {"actions": salvaged_actions}
        raise


def _salvage_partial_actions(json_text):
    actions_key = '"actions"'
    actions_key_index = json_text.find(actions_key)
    if actions_key_index == -1:
        return []

    array_start = json_text.find("[", actions_key_index)
    if array_start == -1:
        return []

    actions = []
    current_start = None
    brace_depth = 0
    in_string = False
    escape_next = False

    for index in range(array_start + 1, len(json_text)):
        char = json_text[index]

        if escape_next:
            escape_next = False
            continue
        if char == "\\" and in_string:
            escape_next = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue

        if char == "{":
            if brace_depth == 0:
                current_start = index
            brace_depth += 1
        elif char == "}":
            if brace_depth > 0:
                brace_depth -= 1
                if brace_depth == 0 and current_start is not None:
                    candidate = json_text[current_start:index + 1]
                    try:
                        actions.append(json.loads(candidate))
                    except json.JSONDecodeError:
                        pass
                    current_start = None
        elif char == "]" and brace_depth == 0:
            break

    return actions


def ask_bailian_for_stacking_order(cubes_info, image_paths=None):
    cube_labels = []
    for i, cube in enumerate(cubes_info):
        color = cube.get("color", f"物块{i + 1}")
        cube_labels.append(f"物块{i + 1}={color}")

    cube_count = len(cubes_info)
    cube_indices_text = ", ".join([f"物块{i + 1}" for i in range(cube_count)])
    json_example = '{"order": [' + ", ".join([f'"物块{i + 1}"' for i in range(cube_count)]) + "]}"

    prompt = f"""你是一个机械臂抓取规划专家。现在有{cube_count}个物块需要被抓取并堆叠在一起。

物块编号与颜色对应关系：
{", ".join(cube_labels)}

请根据三张场景截图中各物块的外观、形状、相对大小和顶部/底部特征，判断最稳定的堆叠顺序。

规则：
1. 底部更宽、更稳、顶部更平整、承托能力更强的物块更适合放在下层。
2. 顶部尖、顶部斜或顶部不平整的物块应尽量放在上层。
3. 第一个抓取的物块会放在最下面，最后一个抓取的物块会放在最上面。
4. 必须严格使用给定的物块编号，不要创造新编号。

请只返回 JSON，格式如下：
{json_example}

其中物块编号是指 {cube_indices_text}。"""

    user_content = [{"type": "text", "text": prompt}]
    for image_path in image_paths or []:
        image_data_url = encode_image_to_data_url(image_path)
        if image_data_url:
            user_content.append({"type": "image_url", "image_url": {"url": image_data_url}})

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {BAILIAN_API_KEY}",
    }
    data = {
        "model": "qwen-vl-max-latest",
        "messages": [
            {"role": "system", "content": "你是机械臂抓取规划专家，请结合图像直接返回 JSON 结果。"},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.1,
        "max_tokens": 160,
    }

    try:
        response = requests.post(BAILIAN_API_URL, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        result = response.json()
        content = result["choices"][0]["message"]["content"]

        print("\n" + "=" * 50)
        print("【阿里云百炼 / Qwen-VL 原始返回内容】")
        print(content)
        print("=" * 50)

        order_result = _load_json_lenient(content)
        order = order_result["order"]
        order_indices = []
        for item in order:
            if isinstance(item, int):
                order_indices.append(item - 1)
            elif isinstance(item, str):
                num = int("".join(filter(str.isdigit, item)))
                order_indices.append(num - 1)

        order_indices = enforce_stacking_constraints(order_indices, cubes_info)
        print(f"解析后的抓取顺序索引: {order_indices}")
        return order_indices
    except Exception as exc:
        print(f"调用阿里云百炼 / Qwen-VL API 失败: {exc}")
        default_order = sorted(
            range(len(cubes_info)),
            key=lambda i: (cubes_info[i].get("top_only", False), -cubes_info[i].get("volume", 0.0)),
        )
        return enforce_stacking_constraints(default_order, cubes_info)


def ask_bailian_for_stack_success(image_paths=None, expected_count=None):
    count_text = f"{expected_count}个物块" if expected_count is not None else "这些物块"
    prompt = f"""你是一个机械臂堆叠结果验收助手。

我会提供当前堆叠完成后的场景图片，请判断 {count_text} 是否已经成功堆叠。

判定标准：
1. 主要关注目标物块是否形成明显的竖向堆叠，而不是散落在桌面。
2. 如果大部分物块已经叠在一起且整体稳定，没有明显倒塌，判定为成功。
3. 如果物块散落、明显滑落、倒塌，或者没有形成堆叠，判定为失败。
4. 只根据图片判断，不要补充额外假设。

请只返回 JSON，格式如下：
{{"success": true, "reason": "一句简短中文说明"}}
"""

    user_content = [{"type": "text", "text": prompt}]
    for image_path in image_paths or []:
        image_data_url = encode_image_to_data_url(image_path)
        if image_data_url:
            user_content.append({"type": "image_url", "image_url": {"url": image_data_url}})

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {BAILIAN_API_KEY}",
    }
    data = {
        "model": "qwen-vl-max-latest",
        "messages": [
            {"role": "system", "content": "你是机械臂堆叠结果验收助手，请直接返回 JSON 结果。"},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.1,
        "max_tokens": 120,
    }

    try:
        response = requests.post(BAILIAN_API_URL, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        result = response.json()
        content = result["choices"][0]["message"]["content"]

        print("\n" + "=" * 50)
        print("【阿里云百炼 / Qwen-VL 堆叠验收原始返回】")
        print(content)
        print("=" * 50)

        verdict = _load_json_lenient(content)
        success = bool(verdict.get("success", False))
        reason = verdict.get("reason", "未提供原因")
        return success, reason
    except Exception as exc:
        print(f"调用阿里云百炼 / Qwen-VL 堆叠验收失败: {exc}")
        return False, f"模型验收失败: {exc}"


def ask_bailian_for_pick_place_actions(cubes_info, image_paths=None, stack_target=None, structure_mode="single_column"):
    stack_target = stack_target or {"x": 0.5, "y": 0.0, "z": 0.0}
    structure_mode = structure_mode or "single_column"
    triangle_count = sum(1 for cube in cubes_info if cube.get("is_triangle"))
    object_lines = []
    for cube in cubes_info:
        grasp_pose = cube.get("default_grasp_pose", {})
        object_lines.append(
            f"物块{cube['index'] + 1}: "
            f"颜色={cube.get('color', '未知')}, "
            f"形状={cube.get('shape', '未知')}, "
            f"位置=({cube['position'][0]:.3f}, {cube['position'][1]:.3f}), "
            f"底面=({cube.get('footprint_x', 0.0):.4f}m x {cube.get('footprint_y', 0.0):.4f}m), "
            f"底面积={cube.get('footprint_area', 0.0):.6f}m^2, "
            f"高度={cube.get('height', 0.0):.4f}m, "
            f"细长比={cube.get('slenderness_ratio', 0.0):.3f}, "
            f"默认抓取候选=(x={grasp_pose.get('x', 0.0):.3f}, y={grasp_pose.get('y', 0.0):.3f}, "
            f"z={grasp_pose.get('z', 0.0):.3f}, yaw={grasp_pose.get('yaw', 0.0):.3f}, "
            f"width={grasp_pose.get('width', 0.05):.3f})"
        )

    structure_rules = {
        "single_column": "这是常规单柱堆叠任务。优先围绕同一个堆叠中心逐层竖直堆叠。",
        "long_bar_pair": "这是实验五。场景中只有两个细长长方体和一个正方体。两个细长长方体应放在下层，位于正方体堆叠点的前后两侧，并使用相同的 layer_index；剩下的一个正方体放在上层 center 位置。",
        "triangle_pair_top": "这是实验六。只有两个三角体需要并排放置：它们应被规划到最高层的同一 layer_index，并且 slot 必须分别为 left 和 right，朝向保持平行。其余普通物块保持 center 的单柱堆叠即可。",
    }
    extra_structure_rule = ""
    if structure_mode == "single_column" and triangle_count == 1:
        extra_structure_rule = (
            "特别注意：当前是常规三角体实验，场景中只有一个三角体。"
            "这个三角体必须独自固定放在整个堆叠结构的最顶层，"
            "它的 layer_index 必须是所有物块里最大的，slot 必须写 center，"
            "不能与任何其他物块同层，也不能放在中间层或底层。"
        )
    json_example = """{
  "actions": [
    {
      "target_object": "物块1",
      "grasp_pose": {"x": 0.10, "y": -0.02, "z": 0.03, "yaw": 0.0, "width": 0.05},
      "layer_index": 0,
      "slot": "center",
      "place_pose": {"x": 0.50, "y": 0.00, "z": 0.00},
      "reason": "适合做底层"
    }
  ]
}"""

    prompt = f"""你是一个机械臂抓取与堆叠规划专家。

现在有 {len(cubes_info)} 个物块，你需要直接给出“抓谁、怎么抓、放到哪里”的动作计划。

场景中物块信息如下：
{os.linesep.join(object_lines)}

我还会给你三张场景图：俯视图、左侧视图、右侧视图。

堆叠目标点建议为：
place_pose = (x={stack_target['x']:.3f}, y={stack_target['y']:.3f}, z={stack_target['z']:.3f})

任务结构要求：
{structure_rules.get(structure_mode, structure_rules['single_column'])}
{extra_structure_rule}

规则：
1. 每个物块只能出现一次。
2. 第一个 action 放在最底层，最后一个 action 放在最上层。
3. 请优先使用我提供的默认抓取候选，只在必要时做小幅调整。
4. 你输出的 grasp_pose 必须是机械臂可以执行的单次抓取位姿。
5. 你必须为每个 action 输出 layer_index 和 slot。
6. slot 只能是 center、left、right、front、back 之一。
7. 当同一层需要成对放两个物块时，必须使用相同的 layer_index，并根据结构要求把两个 slot 写成对应成对槽位。
8. left/right 或 front/back 的物块都要围绕同一个堆叠中心对称，且朝向保持平行。
9. 只能使用给定的物块编号，不要创造新编号。

请只返回 JSON，不要添加其他解释。格式如下：
{json_example}
"""

    user_content = [{"type": "text", "text": prompt}]
    for image_path in image_paths or []:
        image_data_url = encode_image_to_data_url(image_path)
        if image_data_url:
            user_content.append({"type": "image_url", "image_url": {"url": image_data_url}})

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {BAILIAN_API_KEY}",
    }
    data = {
        "model": "qwen-vl-max-latest",
        "messages": [
            {"role": "system", "content": "你是机器人抓取规划专家，请直接返回结构化 JSON 动作计划。"},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.1,
        "max_tokens": 1000,
    }

    try:
        response = requests.post(BAILIAN_API_URL, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        result = response.json()
        content = result["choices"][0]["message"]["content"]

        print("\n" + "=" * 50)
        print("【阿里云百炼 / Qwen-VL 动作计划原始返回】")
        print(content)
        print("=" * 50)

        plan = _load_json_lenient(content)
        actions = plan.get("actions", [])
        if not isinstance(actions, list):
            raise ValueError("actions 字段不是列表")
        return actions
    except Exception as exc:
        print(f"调用阿里云百炼 / Qwen-VL 动作规划失败: {exc}")
        return []
