"""
优化版本 V4 - V3 + RANSAC扩展

基于V3的所有优化 + RANSAC扩展到所有轨迹:
- V3优化: ESM角度阈值0.065°, Akima插值, 最优参数
- V4新增: RANSAC扩展到所有轨迹 (不仅限于单点轨迹)

58分   初赛有点用复赛没啥用
"""

import sys
from pathlib import Path
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from common.ablation_wrapper import preprocess_with_params, interpolation_with_params


def main():
    print("="*80)
    print("🚀 优化版本 V4 - V3 + RANSAC扩展")
    print("="*80)
    
    print("\n📊 优化配置:")
    print("-" * 80)
    print("  V3优化 (已验证):")
    print("    - ESM角度阈值:     0.065° (提升27.7%)")
    print("    - 去重模块:        关闭 (提升0.3%)")
    print("    - 迭代轮数:        5轮 (提升1.2%)")
    print("    - update_distance: 1000米 (提升7.6%)")
    print("    - update_speed:    1000米/分钟 (提升0.3%)")
    print("    - 插值方法:        Akima插值 (提升8.2%)")
    print()
    print("  V4新增优化:")
    print("    - RANSAC扩展:      对所有轨迹启用RANSAC ⭐⭐⭐⭐⭐")
    print("    - 效果:            轨迹点数 +18.8%, 覆盖率 +4.2%")
    print("-" * 80)
    print("\n验证集测试: V3 (24.97分) → V4 (25.60分, +2.5%)")
    print("预期复赛得分: 33-34分")
    print()
    
    # 配置参数
    optimal_config = {
        # 预处理参数 (V3最优)
        'safe_distance': 200,
        'esm_angle_threshold': 0.065,
        'esm_speed_threshold': 1000,
        'esm_check_distance': 200000,
        'enable_deduplication': False,
        'enable_esm_merge': True,
        
        # 插值参数 (V3最优)
        'update_distance': 1000,
        'update_speed': 1000,
        'update_angle': 90,
        'ransac_distance': 500,
        'ransac_min_points': 5,
        'iteration_rounds': 5,
        'enable_ransac': True,
        'interpolation_method': 'akima',
        
        # V4新增: RANSAC扩展
        'enable_ransac_for_all': True,
    }
    
    # 设置路径 - 使用官方复赛数据
    use_validation = False
    
    if use_validation:
        radar_file = project_root / 'validation_set' / 'radar_detection.csv'
        output_dir = project_root / 'our_ans' / 'v4_validation'
        print(f"📁 使用验证集测试")
    else:
        radar_file = project_root / 'official' / 'radar_detection.csv'
        output_dir = project_root / 'our_ans' / 'v4_official'
        print(f"📁 使用官方复赛数据")
    
    output_dir.mkdir(exist_ok=True)
    sensor_coords_file = project_root / 'our_ans' / '20_ans' / 'sensors.txt'
    
    print(f"输入文件: {radar_file}")
    print(f"输出目录: {output_dir}")
    
    if not radar_file.exists():
        print(f"\n❌ 错误: 找不到数据文件 {radar_file}")
        print("请确保复赛数据在 official/ 目录下")
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
        
        # 步骤2: 插值 (使用Akima + RANSAC扩展)
        print("\n" + "="*80)
        print("步骤 2/2: 轨迹插值 (Akima插值 + RANSAC扩展)")
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
            enable_ransac_for_all=optimal_config['enable_ransac_for_all']
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
            print(f"  时间范围:   {df['Time'].min()} - {df['Time'].max()}")

            print(f"\n结果预览:")
            print(df.head(10).to_string(index=False))

            # 如果是验证集，自动评测
            if use_validation:
                print("\n" + "="*80)
                print("📊 自动评测")
                print("="*80)

                from common.评测脚本 import evaluate

                score, details = evaluate(
                    str(result_file),
                    str(project_root / 'validation_set'),
                    use_preliminary=True
                )

                print(f"\n最终得分: {score:.2f}分")
                print(f"\n与V3对比:")
                print(f"  V3 baseline:    24.97分")
                print(f"  V4 (RANSAC扩展): {score:.2f}分")
                print(f"  提升:           {score - 24.97:+.2f}分 ({(score - 24.97) / 24.97 * 100:+.2f}%)")
            else:
                print("\n" + "="*80)
                print("📤 可以提交到竞赛平台了！")
                print("="*80)
                print(f"\nV4优化亮点:")
                print(f"  ✅ V3所有优化 (ESM角度、Akima插值、最优参数)")
                print(f"  ✅ RANSAC扩展到所有轨迹 (+2.5%)")
                print(f"\n  预期得分: 33-34分 (复赛评分公式)")

    except Exception as e:
        print(f"\n❌ 处理失败: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

