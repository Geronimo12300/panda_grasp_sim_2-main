"""
大模型规划模块测试脚本

测试目标：验证大模型输出格式和参数合理性

测试方法：
- 构造标准场景输入（物体信息+图像）
- 调用大模型API，记录输出结果
- 验证JSON格式正确性和参数范围

测试用例：
- 用例1：标准单柱堆叠场景
- 用例2：细长长方体配对场景
- 用例3：三角体配对场景

评价指标：
- JSON格式正确率：正确解析次数/总调用次数
- 参数合理性：参数在约束范围内的比例
- 规划完整性：覆盖所有物体的比例

用法:
    python test_llm_planning.py
    python test_llm_planning.py --provider kimi
    python test_llm_planning.py --trials 3
"""

import os
import sys
import json
import argparse
import time
import numpy as np
from datetime import datetime

# 设置控制台编码为UTF-8，避免Windows GBK编码错误
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm_client import (
    ask_bailian_for_pick_place_actions,
    set_llm_provider,
    get_current_llm_info,
    LLM_PROVIDERS
)


def create_mock_cubes_info(num_objects=5, scenario="standard"):
    """
    创建模拟的物体信息
    
    参数:
        num_objects: 物体数量
        scenario: 场景类型 (standard, long_bar_pair, triangle_pair_top)
    """
    colors = ["红色", "绿色", "蓝色", "黄色", "紫色"]
    
    cubes_info = []
    
    if scenario == "standard":
        shapes = ["cube", "cube", "cylinder", "cylinder", "cube"]
        for i in range(num_objects):
            shape = shapes[i % len(shapes)]
            if shape == "cube":
                size = np.random.uniform(0.04, 0.05)
                cubes_info.append({
                    "index": i,
                    "color": colors[i % len(colors)],
                    "shape": "正方体",
                    "position": [
                        np.random.uniform(-0.15, 0.15),
                        np.random.uniform(-0.12, 0.12),
                        0.02
                    ],
                    "height": size,
                    "footprint_x": size,
                    "footprint_y": size,
                    "footprint_area": size * size,
                    "volume": size ** 3,
                    "is_triangle": False,
                    "default_grasp_pose": {
                        "x": np.random.uniform(-0.15, 0.15),
                        "y": np.random.uniform(-0.12, 0.12),
                        "z": 0.02,
                        "yaw": 0.0,
                        "width": size + 0.01
                    }
                })
            else:
                radius = np.random.uniform(0.02, 0.035)
                height = np.random.uniform(0.04, 0.06)
                cubes_info.append({
                    "index": i,
                    "color": colors[i % len(colors)],
                    "shape": "圆柱体",
                    "position": [
                        np.random.uniform(-0.15, 0.15),
                        np.random.uniform(-0.12, 0.12),
                        0.02
                    ],
                    "height": height,
                    "footprint_x": radius * 2,
                    "footprint_y": radius * 2,
                    "footprint_area": np.pi * radius ** 2,
                    "volume": np.pi * radius ** 2 * height,
                    "is_triangle": False,
                    "default_grasp_pose": {
                        "x": np.random.uniform(-0.15, 0.15),
                        "y": np.random.uniform(-0.12, 0.12),
                        "z": 0.02,
                        "yaw": 0.0,
                        "width": radius * 2 + 0.01
                    }
                })
    
    elif scenario == "long_bar_pair":
        cubes_info = [
            {
                "index": 0,
                "color": "红色",
                "shape": "正方体",
                "position": [0.0, 0.0, 0.02],
                "height": 0.045,
                "footprint_x": 0.045,
                "footprint_y": 0.045,
                "footprint_area": 0.045 * 0.045,
                "volume": 0.045 ** 3,
                "is_triangle": False,
                "slenderness_ratio": 1.0,
                "default_grasp_pose": {"x": 0.0, "y": 0.0, "z": 0.02, "yaw": 0.0, "width": 0.05}
            },
            {
                "index": 1,
                "color": "绿色",
                "shape": "细长长方体",
                "position": [-0.08, 0.05, 0.02],
                "height": 0.04,
                "footprint_x": 0.08,
                "footprint_y": 0.025,
                "footprint_area": 0.08 * 0.025,
                "volume": 0.08 * 0.025 * 0.04,
                "is_triangle": False,
                "slenderness_ratio": 3.2,
                "default_grasp_pose": {"x": -0.08, "y": 0.05, "z": 0.02, "yaw": 0.0, "width": 0.03}
            },
            {
                "index": 2,
                "color": "蓝色",
                "shape": "细长长方体",
                "position": [0.08, -0.05, 0.02],
                "height": 0.04,
                "footprint_x": 0.08,
                "footprint_y": 0.025,
                "footprint_area": 0.08 * 0.025,
                "volume": 0.08 * 0.025 * 0.04,
                "is_triangle": False,
                "slenderness_ratio": 3.2,
                "default_grasp_pose": {"x": 0.08, "y": -0.05, "z": 0.02, "yaw": 0.0, "width": 0.03}
            }
        ]
    
    elif scenario == "triangle_pair_top":
        cubes_info = [
            {
                "index": 0,
                "color": "红色",
                "shape": "正方体",
                "position": [0.0, 0.0, 0.02],
                "height": 0.045,
                "footprint_x": 0.045,
                "footprint_y": 0.045,
                "footprint_area": 0.045 * 0.045,
                "volume": 0.045 ** 3,
                "is_triangle": False,
                "default_grasp_pose": {"x": 0.0, "y": 0.0, "z": 0.02, "yaw": 0.0, "width": 0.05}
            },
            {
                "index": 1,
                "color": "绿色",
                "shape": "三角体",
                "position": [-0.06, 0.08, 0.02],
                "height": 0.05,
                "footprint_x": 0.05,
                "footprint_y": 0.05,
                "footprint_area": 0.5 * 0.05 * 0.05,
                "volume": 0.5 * 0.05 * 0.05 * 0.05,
                "is_triangle": True,
                "top_only": True,
                "default_grasp_pose": {"x": -0.06, "y": 0.08, "z": 0.02, "yaw": 0.0, "width": 0.04}
            },
            {
                "index": 2,
                "color": "蓝色",
                "shape": "三角体",
                "position": [0.06, -0.08, 0.02],
                "height": 0.05,
                "footprint_x": 0.05,
                "footprint_y": 0.05,
                "footprint_area": 0.5 * 0.05 * 0.05,
                "volume": 0.5 * 0.05 * 0.05 * 0.05,
                "is_triangle": True,
                "top_only": True,
                "default_grasp_pose": {"x": 0.06, "y": -0.08, "z": 0.02, "yaw": 0.0, "width": 0.04}
            }
        ]
    
    return cubes_info


def create_mock_images(output_dir, scenario="standard"):
    """创建模拟图像文件（空白图像用于测试）"""
    import cv2
    
    os.makedirs(output_dir, exist_ok=True)
    
    img_paths = []
    view_names = ["top", "left", "right"]
    
    for view in view_names:
        img_path = os.path.join(output_dir, f"camera_rgb_{view}.png")
        
        img = np.ones((480, 640, 3), dtype=np.uint8) * 200
        
        cv2.putText(img, f"Mock Image - {scenario} - {view}", (100, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        cv2.imwrite(img_path, img)
        img_paths.append(img_path)
    
    return img_paths


def validate_action(action, cubes_info, scenario="standard"):
    """
    验证单个action的参数合理性
    
    返回:
        (is_valid, errors)
    """
    errors = []
    
    required_fields = ["target_object", "grasp_pose", "layer_index", "slot", "place_pose"]
    for field in required_fields:
        if field not in action:
            errors.append(f"缺少必需字段: {field}")
    
    if "grasp_pose" in action:
        grasp = action["grasp_pose"]
        grasp_required = ["x", "y", "z", "yaw", "width"]
        for field in grasp_required:
            if field not in grasp:
                errors.append(f"grasp_pose缺少字段: {field}")
        
        if "width" in grasp:
            if not (0.02 <= grasp["width"] <= 0.08):
                errors.append(f"grasp_pose.width={grasp['width']:.4f} 超出范围 [0.02, 0.08]")
        
        if "z" in grasp:
            if not (0.0 <= grasp["z"] <= 0.1):
                errors.append(f"grasp_pose.z={grasp['z']:.4f} 超出范围 [0.0, 0.1]")
    
    if "place_pose" in action:
        place = action["place_pose"]
        place_required = ["x", "y", "z"]
        for field in place_required:
            if field not in place:
                errors.append(f"place_pose缺少字段: {field}")
    
    if "slot" in action:
        valid_slots = ["center", "left", "right", "front", "back"]
        if action["slot"] not in valid_slots:
            errors.append(f"slot={action['slot']} 不是有效值 {valid_slots}")
    
    if "layer_index" in action:
        if not isinstance(action["layer_index"], int) or action["layer_index"] < 0:
            errors.append(f"layer_index={action['layer_index']} 不是有效的非负整数")
    
    return len(errors) == 0, errors


def validate_actions(actions, cubes_info, scenario="standard"):
    """
    验证actions列表的完整性和合理性
    
    返回:
        {
            "json_valid": bool,
            "coverage": float,
            "param_valid_ratio": float,
            "errors": list,
            "warnings": list
        }
    """
    result = {
        "json_valid": True,
        "coverage": 0.0,
        "param_valid_ratio": 0.0,
        "errors": [],
        "warnings": [],
        "action_errors": []
    }
    
    if not isinstance(actions, list):
        result["json_valid"] = False
        result["errors"].append("actions不是列表")
        return result
    
    if len(actions) == 0:
        result["errors"].append("actions为空")
        return result
    
    num_objects = len(cubes_info)
    covered_objects = set()
    valid_action_count = 0
    
    for i, action in enumerate(actions):
        is_valid, errors = validate_action(action, cubes_info, scenario)
        if is_valid:
            valid_action_count += 1
        else:
            result["action_errors"].append({
                "action_index": i,
                "errors": errors
            })
        
        if "target_object" in action:
            try:
                target = action["target_object"]
                if isinstance(target, str) and "物块" in target:
                    num = int(target.replace("物块", ""))
                    covered_objects.add(num - 1)
                elif isinstance(target, int):
                    covered_objects.add(target - 1)
            except:
                pass
    
    result["coverage"] = len(covered_objects) / num_objects if num_objects > 0 else 0
    result["param_valid_ratio"] = valid_action_count / len(actions) if actions else 0
    
    if len(covered_objects) < num_objects:
        missing = [i + 1 for i in range(num_objects) if i not in covered_objects]
        result["warnings"].append(f"未覆盖物块: {missing}")
    
    if len(covered_objects) > num_objects:
        result["warnings"].append(f"覆盖物块数({len(covered_objects)})超过实际数({num_objects})")
    
    if scenario == "long_bar_pair":
        slots = [a.get("slot") for a in actions]
        if "left" in slots and "right" in slots:
            pass
        else:
            result["warnings"].append("细长长方体场景应使用left/right配对")
    
    elif scenario == "triangle_pair_top":
        triangle_actions = []
        for a in actions:
            target = a.get("target_object", "")
            idx = None
            if isinstance(target, str) and "物块" in target:
                idx = int(target.replace("物块", "")) - 1
            elif isinstance(target, int):
                idx = target - 1
            if idx is not None and idx < len(cubes_info) and cubes_info[idx].get("is_triangle"):
                triangle_actions.append(a)
        
        if len(triangle_actions) == 2:
            slots = [a.get("slot") for a in triangle_actions]
            if "left" in slots and "right" in slots:
                pass
            else:
                result["warnings"].append("三角体应使用left/right配对")
    
    return result


def run_single_test(scenario, trial_index, output_dir):
    """运行单次测试"""
    np.random.seed(42 + trial_index * 100)
    
    cubes_info = create_mock_cubes_info(num_objects=5 if scenario == "standard" else 3, scenario=scenario)
    
    img_dir = os.path.join(output_dir, f"test_{scenario}_{trial_index}")
    img_paths = create_mock_images(img_dir, scenario)
    
    stack_target = {"x": 0.5, "y": 0.0, "z": 0.0}
    
    structure_mode = "single_column"
    if scenario == "long_bar_pair":
        structure_mode = "long_bar_pair"
    elif scenario == "triangle_pair_top":
        structure_mode = "triangle_pair_top"
    
    start_time = time.time()
    
    try:
        actions = ask_bailian_for_pick_place_actions(
            cubes_info,
            image_paths=img_paths,
            stack_target=stack_target,
            structure_mode=structure_mode
        )
        
        elapsed_time = time.time() - start_time
        
        if actions is None:
            actions = []
        
        validation = validate_actions(actions, cubes_info, scenario)
        
        return {
            "scenario": scenario,
            "trial": trial_index,
            "success": True,
            "elapsed_time": elapsed_time,
            "num_actions": len(actions),
            "validation": validation,
            "actions": actions
        }
    
    except Exception as e:
        elapsed_time = time.time() - start_time
        return {
            "scenario": scenario,
            "trial": trial_index,
            "success": False,
            "error": str(e),
            "elapsed_time": elapsed_time,
            "num_actions": 0,
            "validation": {
                "json_valid": False,
                "coverage": 0,
                "param_valid_ratio": 0,
                "errors": [str(e)]
            }
        }


def print_scenario_report(results, scenario_name):
    """打印单个场景的测试报告"""
    print(f"\n{'='*60}")
    print(f"场景: {scenario_name}")
    print(f"{'='*60}")
    
    success_count = sum(1 for r in results if r["success"])
    json_valid_count = sum(1 for r in results if r["validation"]["json_valid"])
    
    coverages = [r["validation"]["coverage"] for r in results if r["success"]]
    param_ratios = [r["validation"]["param_valid_ratio"] for r in results if r["success"]]
    elapsed_times = [r["elapsed_time"] for r in results]
    
    print(f"\n【基本统计】")
    print(f"  测试次数: {len(results)}")
    print(f"  API调用成功: {success_count}/{len(results)} ({success_count/len(results)*100:.1f}%)")
    print(f"  JSON格式正确: {json_valid_count}/{len(results)} ({json_valid_count/len(results)*100:.1f}%)")
    
    if coverages:
        print(f"\n【规划完整性】")
        print(f"  平均覆盖率: {np.mean(coverages)*100:.1f}%")
        print(f"  完全覆盖次数: {sum(1 for c in coverages if c >= 1.0)}/{len(coverages)}")
    
    if param_ratios:
        print(f"\n【参数合理性】")
        print(f"  平均参数合理率: {np.mean(param_ratios)*100:.1f}%")
        print(f"  全部合理次数: {sum(1 for p in param_ratios if p >= 1.0)}/{len(param_ratios)}")
    
    print(f"\n【响应时间】")
    print(f"  平均响应时间: {np.mean(elapsed_times):.2f}s")
    print(f"  最大响应时间: {np.max(elapsed_times):.2f}s")
    print(f"  最小响应时间: {np.min(elapsed_times):.2f}s")
    
    error_count = sum(len(r["validation"].get("errors", [])) for r in results)
    warning_count = sum(len(r["validation"].get("warnings", [])) for r in results)
    
    if error_count > 0 or warning_count > 0:
        print(f"\n【问题统计】")
        print(f"  错误总数: {error_count}")
        print(f"  警告总数: {warning_count}")


def main():
    parser = argparse.ArgumentParser(description="大模型规划模块测试")
    parser.add_argument("--provider", type=str, default=None, 
                        choices=list(LLM_PROVIDERS.keys()),
                        help="LLM提供商")
    parser.add_argument("--trials", type=int, default=3, help="每个场景测试次数")
    parser.add_argument("--output", type=str, default="test_reports", help="输出目录")
    
    args = parser.parse_args()
    
    if args.provider:
        set_llm_provider(args.provider)
    
    current_info = get_current_llm_info()
    
    print("="*60)
    print("大模型规划模块测试")
    print("="*60)
    print(f"当前模型: {current_info['provider_name']}")
    print(f"模型: {current_info['model']}")
    print(f"每个场景测试次数: {args.trials}")
    print("="*60)
    
    scenarios = [
        ("standard", "标准单柱堆叠场景"),
        ("long_bar_pair", "细长长方体配对场景"),
        ("triangle_pair_top", "三角体配对场景")
    ]
    
    all_results = {}
    
    for scenario_id, scenario_name in scenarios:
        print(f"\n>>> 测试场景: {scenario_name}")
        
        results = []
        for i in range(args.trials):
            print(f"  [{i+1}/{args.trials}] 调用API...", end=" ", flush=True)
            result = run_single_test(scenario_id, i + 1, args.output)
            results.append(result)
            
            if result["success"]:
                v = result["validation"]
                print(f"成功 - 覆盖率: {v['coverage']*100:.0f}%, 参数合理: {v['param_valid_ratio']*100:.0f}%, 耗时: {result['elapsed_time']:.1f}s")
            else:
                print(f"失败 - {result.get('error', '未知错误')}")
        
        all_results[scenario_id] = results
        print_scenario_report(results, scenario_name)
    
    print("\n" + "="*60)
    print("总体统计")
    print("="*60)
    
    total_results = []
    for results in all_results.values():
        total_results.extend(results)
    
    total_success = sum(1 for r in total_results if r["success"])
    total_json_valid = sum(1 for r in total_results if r["validation"]["json_valid"])
    total_coverages = [r["validation"]["coverage"] for r in total_results if r["success"]]
    total_param_ratios = [r["validation"]["param_valid_ratio"] for r in total_results if r["success"]]
    
    print(f"\n总测试次数: {len(total_results)}")
    print(f"API调用成功率: {total_success/len(total_results)*100:.1f}%")
    print(f"JSON格式正确率: {total_json_valid/len(total_results)*100:.1f}%")
    
    if total_coverages:
        print(f"平均覆盖率: {np.mean(total_coverages)*100:.1f}%")
    if total_param_ratios:
        print(f"平均参数合理率: {np.mean(total_param_ratios)*100:.1f}%")
    
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "model": current_info,
        "trials_per_scenario": args.trials,
        "summary": {
            "total_tests": len(total_results),
            "api_success_rate": total_success/len(total_results),
            "json_valid_rate": total_json_valid/len(total_results),
            "avg_coverage": float(np.mean(total_coverages)) if total_coverages else 0,
            "avg_param_valid_ratio": float(np.mean(total_param_ratios)) if total_param_ratios else 0,
        },
        "scenarios": {
            scenario_id: {
                "name": name,
                "results": [
                    {
                        "trial": r["trial"],
                        "success": r["success"],
                        "elapsed_time": r["elapsed_time"],
                        "coverage": r["validation"]["coverage"],
                        "param_valid_ratio": r["validation"]["param_valid_ratio"],
                        "errors": r["validation"].get("errors", []),
                        "warnings": r["validation"].get("warnings", [])
                    }
                    for r in results
                ]
            }
            for (scenario_id, name), results in zip(scenarios, all_results.values())
        }
    }
    
    output_path = os.path.join(args.output, "llm_test_results.json")
    os.makedirs(args.output, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n详细结果已保存到: {output_path}")


if __name__ == "__main__":
    main()
