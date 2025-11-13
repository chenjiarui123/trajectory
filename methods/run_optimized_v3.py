"""
优化版本 v3 - 应用Akima插值

基于插值方法对比实验结果:
- Akima插值相比原始样条: +8.2% (36.68 vs 33.89)
- 结合v2的参数优化: update_distance=1000 (+7.6%), update_speed=1000 (+0.3%)
- 总提升: 约 +40% (相比最初baseline)
//59分
"""

import sys
from pathlib import Path
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from common.ablation_wrapper import preprocess_with_params, interpolation_with_params


def main():
    print("="*80)
    print("🚀 优化版本 v3 - Akima插值 + 最优参数")
    print("="*80)
    
    print("\n📊 优化配置:")
    print("-" * 80)
    print("  预处理参数 (已优化):")
    print("    - ESM角度阈值:     0.065° (提升27.7%)")
    print("    - 去重模块:        关闭 (提升0.3%)")
    print("    - 迭代轮数:        5轮 (提升1.2%)")
    print()
    print("  插值参数 (已优化):")
    print("    - update_distance: 1000米 (提升7.6%) ⭐⭐⭐")
    print("    - update_speed:    1000米/分钟 (提升0.3%) ⭐")
    print("    - update_angle:    90度 (保持)")
    print()
    print("  插值方法 (新优化):")
    print("    - 方法:            Akima插值 (提升8.2%) ⭐⭐⭐⭐⭐")
    print("    - 优势:            局部插值，对噪声鲁棒，无振荡")
    print("-" * 80)
    print("\n预期总提升: ~40% (从22.83到约32分)")
    print()
    
    # 配置参数
    optimal_config = {
        # 预处理参数 (已验证最优)
        'safe_distance': 200,
        'esm_angle_threshold': 0.065,      # ⭐⭐⭐ 最关键
        'esm_speed_threshold': 1000,
        'esm_check_distance': 200000,
        'enable_deduplication': False,     # ⭐ 关闭去重
        'enable_esm_merge': True,
        
        # 插值参数 (已优化)
        'update_distance': 1000,           # ⭐⭐⭐ 从600提升到1000
        'update_speed': 1000,              # ⭐ 从800提升到1000
        'update_angle': 90,
        'ransac_distance': 500,
        'ransac_min_points': 5,
        'iteration_rounds': 5,             # ⭐ 已优化
        'enable_ransac': True,
        
        # 插值方法 (新增)
        'interpolation_method': 'akima',   # ⭐⭐⭐⭐⭐ Akima插值
    }
    
    # 设置路径
    radar_file = project_root / 'official' / 'radar_detection.csv'
    sensor_coords_file = project_root / 'our_ans' / '20_ans' / 'sensors.txt'
    output_dir = project_root / 'our_ans' / 'optimized_v3_results'
    output_dir.mkdir(exist_ok=True)
    
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
        print("正在处理...")
        
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
        
        # 步骤2: 插值 (使用Akima方法)
        print("\n" + "="*80)
        print("步骤 2/2: 轨迹插值 (Akima插值)")
        print("="*80)
        print("正在处理...")
        
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
            interpolation_method=optimal_config['interpolation_method']
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
            
            print("\n" + "="*80)
            print("📤 可以提交到竞赛平台了！")
            print("="*80)
            print(f"\n优化亮点:")
            print(f"  ✅ ESM角度阈值优化 (+27.7%)")
            print(f"  ✅ 关闭去重模块 (+0.3%)")
            print(f"  ✅ 5轮迭代 (+1.2%)")
            print(f"  ✅ update_distance=1000 (+7.6%)")
            print(f"  ✅ update_speed=1000 (+0.3%)")
            print(f"  ✅ Akima插值 (+8.2%) ⭐ 最新优化")
            print(f"\n  预期总提升: ~40%")
            print(f"  预期得分: 31-32分 (复赛评分公式)")
        
    except Exception as e:
        print(f"\n❌ 处理失败: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

