"""
优化版本 v3.3 - 离群点检测

基于 v3.2，新增插值前的离群点检测

改进点：
- update_distance: 2000 (继承 v3.1)
- enable_kinematic_constraints: True (继承 v3.2)
- enable_outlier_detection: True ⭐ (新增)

预期收益：+2-3 分（相比 v3.2）
"""

import sys
from pathlib import Path
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from common.ablation_wrapper import preprocess_with_params, interpolation_with_params


def main():
    print("="*80)
    print("🚀 优化版本 v3.3 - 离群点检测")
    print("="*80)
    
    print("\n📊 优化配置:")
    print("-" * 80)
    print("  继承 v3.1:")
    print("    - update_distance: 2000")
    print()
    print("  继承 v3.2:")
    print("    - 运动学约束: 启用")
    print()
    print("  新增优化:")
    print("    - 离群点检测: 启用 ⭐⭐⭐")
    print("      · 插值前清理速度异常点")
    print("      · 阈值: 60 节")
    print("-" * 80)
    print("\n预期收益: +2-3 分（相比 v3.2）")
    
    # 配置参数
    optimal_config = {
        # 预处理参数 (继承 v3)
        'safe_distance': 200,
        'esm_angle_threshold': 0.065,
        'esm_speed_threshold': 1000,
        'esm_check_distance': 200000,
        'enable_deduplication': False,
        'enable_esm_merge': True,
        
        # 插值参数
        'update_distance': 2000,           # 继承 v3.1
        'update_speed': 1000,
        'update_angle': 90,
        'ransac_distance': 500,
        'ransac_min_points': 5,
        'iteration_rounds': 5,
        'enable_ransac': True,
        'interpolation_method': 'akima',
        
        # v3.2 继承
        'enable_kinematic_constraints': True,
        
        # v3.3 新增
        'enable_outlier_detection': True,  # ⭐ 离群点检测
    }
    
    # 设置路径
    radar_file = project_root / 'official' / 'radar_detection.csv'
    sensor_coords_file = project_root / 'our_ans' / '20_ans' / 'sensors.txt'
    output_dir = project_root / 'our_ans' / 'optimized_v3_3_results'
    output_dir.mkdir(exist_ok=True)
    
    print(f"\n输入文件: {radar_file}")
    print(f"输出目录: {output_dir}")
    
    if not radar_file.exists():
        print(f"\n❌ 错误: 找不到数据文件 {radar_file}")
        return
    
    try:
        # 步骤1: 预处理
        print("\n" + "="*80)
        print("步骤 1/2: 预处理")
        print("="*80)
        
        preprocess_with_params(
            str(radar_file),
            str(sensor_coords_file),
            str(output_dir),
            safe_distance=optimal_config['safe_distance'],
            esm_angle_threshold=optimal_config['esm_angle_threshold'],
            esm_speed_threshold=optimal_config['esm_speed_threshold'],
            esm_check_distance=optimal_config['esm_check_distance'],
            enable_deduplication=optimal_config['enable_deduplication'],
            enable_esm_merge=optimal_config['enable_esm_merge']
        )
        
        print("✓ 预处理完成")
        
        # 步骤2: 插值
        print("\n" + "="*80)
        print("步骤 2/2: 轨迹插值 (运动学约束 + 离群点检测)")
        print("="*80)
        
        located_ESM_file = output_dir / 'located_ESM_points_updated.csv'
        radar_points_file = output_dir / 'deduplacate_radar_points_updated.csv'
        
        interpolation_with_params(
            str(located_ESM_file),
            str(radar_points_file),
            str(output_dir),
            update_distance=optimal_config['update_distance'],
            update_speed=optimal_config['update_speed'],
            update_angle=optimal_config['update_angle'],
            ransac_distance=optimal_config['ransac_distance'],
            ransac_min_points=optimal_config['ransac_min_points'],
            iteration_rounds=optimal_config['iteration_rounds'],
            enable_ransac=optimal_config['enable_ransac'],
            interpolation_method=optimal_config['interpolation_method'],
            enable_kinematic_constraints=optimal_config['enable_kinematic_constraints'],
            enable_outlier_detection=optimal_config['enable_outlier_detection']  # ⭐
        )
        
        print("✓ 插值完成")
        
        # 检查结果
        result_file = output_dir / 'results.csv'
        
        print("\n" + "="*80)
        print("✅ 处理完成！")
        print("="*80)
        print(f"\n结果文件: {result_file}")
        
        if result_file.exists():
            df = pd.read_csv(result_file)
            print(f"\n结果统计:")
            print(f"  总轨迹点数: {len(df):,}")
            print(f"  船只数量:   {df['ID'].nunique():,}")
        
    except Exception as e:
        print(f"\n❌ 处理失败: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

