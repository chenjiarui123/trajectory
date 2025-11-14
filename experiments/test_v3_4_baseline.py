"""
测试 v3.4 在 validation_set 上的真实 baseline 得分

配置：
- update_distance = 2000
- update_speed = 1000
- update_angle = 60
- enable_kinematic_constraints = True
- enable_outlier_detection = True (虽然可能没实现)
"""

import sys
from pathlib import Path
import time

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from common.ablation_wrapper import preprocess_with_params, interpolation_with_params

# 导入评测函数
sys.path.insert(0, str(project_root / 'common'))
from 评测脚本 import evaluate


def main():
    print("="*80)
    print("🧪 测试 v3.4 在 validation_set 上的真实 baseline")
    print("="*80)
    
    # v3.4 配置
    config = {
        'safe_distance': 200,
        'esm_angle_threshold': 0.065,
        'esm_speed_threshold': 1000,
        'esm_check_distance': 200000,
        'enable_deduplication': False,
        'enable_esm_merge': True,
        'update_distance': 2000,
        'update_speed': 1000,
        'update_angle': 60,
        'ransac_distance': 500,
        'ransac_min_points': 5,
        'iteration_rounds': 5,
        'enable_ransac': True,
        'interpolation_method': 'akima',
        'enable_kinematic_constraints': True,
        'enable_outlier_detection': True,
    }
    
    # 数据路径
    validation_dir = project_root / 'validation_set'
    radar_file = validation_dir / 'radar_detection.csv'
    sensor_coords = project_root / 'our_ans' / '20_ans' / 'sensors.txt'
    output_dir = project_root / 'experiments' / 'v3_4_baseline_test'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n配置:")
    print(f"  update_distance = {config['update_distance']}")
    print(f"  update_speed = {config['update_speed']}")
    print(f"  update_angle = {config['update_angle']}")
    print(f"  enable_kinematic_constraints = {config['enable_kinematic_constraints']}")
    print(f"  enable_outlier_detection = {config['enable_outlier_detection']}")
    
    start_time = time.time()
    
    # 步骤1: 预处理
    print(f"\n步骤 1/3: 预处理...")
    preprocess_with_params(
        str(radar_file),
        str(sensor_coords),
        str(output_dir),
        safe_distance=config['safe_distance'],
        esm_angle_threshold=config['esm_angle_threshold'],
        esm_speed_threshold=config['esm_speed_threshold'],
        esm_check_distance=config['esm_check_distance'],
        enable_deduplication=config['enable_deduplication'],
        enable_esm_merge=config['enable_esm_merge']
    )
    
    # 步骤2: 插值
    print(f"步骤 2/3: 插值...")
    located_ESM_file = output_dir / 'located_ESM_points_updated.csv'
    radar_points_file = output_dir / 'deduplacate_radar_points_updated.csv'
    
    interpolation_with_params(
        str(located_ESM_file),
        str(radar_points_file),
        str(output_dir),
        update_distance=config['update_distance'],
        update_speed=config['update_speed'],
        update_angle=config['update_angle'],
        ransac_distance=config['ransac_distance'],
        ransac_min_points=config['ransac_min_points'],
        iteration_rounds=config['iteration_rounds'],
        enable_ransac=config['enable_ransac'],
        interpolation_method=config['interpolation_method'],
        enable_kinematic_constraints=config['enable_kinematic_constraints'],
        enable_outlier_detection=config['enable_outlier_detection']
    )
    
    # 步骤3: 评估
    print(f"步骤 3/3: 评估...")
    result_file = output_dir / 'results.csv'
    
    if not result_file.exists():
        print(f"❌ 结果文件不存在")
        return
    
    score, _ = evaluate(
        str(result_file),
        validation_dir=str(validation_dir),
        use_preliminary=True
    )
    
    elapsed = time.time() - start_time
    
    print(f"\n{'='*80}")
    print(f"📊 v3.4 Baseline 测试结果")
    print(f"{'='*80}")
    print(f"  得分: {score:.4f}")
    print(f"  耗时: {elapsed:.1f} 秒")
    print(f"{'='*80}")
    
    # 对比之前的网格搜索结果
    print(f"\n对比:")
    print(f"  v3.4 (2000, 1000): {score:.4f}")
    print(f"  网格搜索最优 (1900, 1000): 26.1710")
    print(f"  差异: {score - 26.1710:+.4f}")


if __name__ == '__main__':
    main()

