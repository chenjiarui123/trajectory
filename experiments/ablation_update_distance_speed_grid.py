"""
消融实验：update_distance × update_speed 联合网格搜索

基于 v3.4 配置，寻找最优的 (update_distance, update_speed) 组合

测试范围：
- update_distance: [1800, 1900, 2000, 2100, 2200, 2300] (围绕2000精细搜索)
- update_speed: [800, 1000, 1200, 1500, 2000]

数据集：validation_set

运行方式：
    python experiments/ablation_update_distance_speed_grid.py
"""

import sys
from pathlib import Path
import pandas as pd
import time
import itertools

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from common.ablation_wrapper import preprocess_with_params, interpolation_with_params

# 导入评测函数
sys.path.insert(0, str(project_root / 'common'))
from 评测脚本 import evaluate


def run_experiment(update_distance, update_speed, base_config, data_paths):
    """运行单次实验"""
    print(f"\n{'='*80}")
    print(f"测试 update_distance={update_distance}, update_speed={update_speed}")
    print(f"{'='*80}")

    # 创建输出目录
    output_dir = data_paths['output_base'] / f'dist_{update_distance}_speed_{update_speed}'
    output_dir.mkdir(parents=True, exist_ok=True)

    # 检查是否已经完成
    result_file = output_dir / 'results.csv'
    if result_file.exists():
        print(f"⏭️  已完成，跳过")
        try:
            score, _ = evaluate(
                str(result_file),
                validation_dir=str(data_paths['validation_dir']),
                use_preliminary=True
            )
            return {
                'update_distance': update_distance,
                'update_speed': update_speed,
                'score': score,
                'time': 0
            }
        except:
            print(f"⚠️  结果文件损坏，重新运行")
            pass
    
    start_time = time.time()
    
    try:
        # 步骤1: 预处理
        print("步骤 1/3: 预处理...")
        preprocess_with_params(
            str(data_paths['radar_file']),
            str(data_paths['sensor_coords']),
            str(output_dir),
            safe_distance=base_config['safe_distance'],
            esm_angle_threshold=base_config['esm_angle_threshold'],
            esm_speed_threshold=base_config['esm_speed_threshold'],
            esm_check_distance=base_config['esm_check_distance'],
            enable_deduplication=base_config['enable_deduplication'],
            enable_esm_merge=base_config['enable_esm_merge']
        )
        
        # 步骤2: 插值
        print("步骤 2/3: 插值...")
        located_ESM_file = output_dir / 'located_ESM_points_updated.csv'
        radar_points_file = output_dir / 'deduplacate_radar_points_updated.csv'
        
        interpolation_with_params(
            str(located_ESM_file),
            str(radar_points_file),
            str(output_dir),
            update_distance=update_distance,  # ⭐ 变化参数1
            update_speed=update_speed,        # ⭐ 变化参数2
            update_angle=base_config['update_angle'],
            ransac_distance=base_config['ransac_distance'],
            ransac_min_points=base_config['ransac_min_points'],
            iteration_rounds=base_config['iteration_rounds'],
            enable_ransac=base_config['enable_ransac'],
            interpolation_method=base_config['interpolation_method'],
            enable_kinematic_constraints=base_config['enable_kinematic_constraints'],
            enable_outlier_detection=base_config['enable_outlier_detection']
        )
        
        # 步骤3: 评估
        print("步骤 3/3: 评估...")
        result_file = output_dir / 'results.csv'
        
        if not result_file.exists():
            print(f"❌ 结果文件不存在")
            return None
        
        score, _ = evaluate(
            str(result_file),
            validation_dir=str(data_paths['validation_dir']),
            use_preliminary=True
        )
        
        elapsed = time.time() - start_time
        
        print(f"✅ 得分: {score:.4f}, 耗时: {elapsed:.1f}秒")
        
        return {
            'update_distance': update_distance,
            'update_speed': update_speed,
            'score': score,
            'time': elapsed
        }
        
    except Exception as e:
        print(f"❌ 实验失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def main():
    print("="*80)
    print("🔬 消融实验：update_distance × update_speed 联合网格搜索")
    print("="*80)
    
    # 数据路径
    validation_dir = project_root / 'validation_set'
    data_paths = {
        'validation_dir': validation_dir,
        'radar_file': validation_dir / 'radar_detection.csv',
        'sensor_coords': project_root / 'our_ans' / '20_ans' / 'sensors.txt',
        'output_base': project_root / 'experiments' / 'ablation_results' / 'distance_speed_grid'
    }
    
    if not data_paths['radar_file'].exists():
        print(f"❌ 找不到验证集数据: {data_paths['radar_file']}")
        return
    
    # v3.4 的基础配置（除了 update_distance 和 update_speed）
    base_config = {
        'safe_distance': 200,
        'esm_angle_threshold': 0.065,
        'esm_speed_threshold': 1000,
        'esm_check_distance': 200000,
        'enable_deduplication': False,
        'enable_esm_merge': True,
        'update_angle': 60,  # v3.4 优化后的值
        'ransac_distance': 500,
        'ransac_min_points': 5,
        'iteration_rounds': 5,
        'enable_ransac': True,
        'interpolation_method': 'akima',
        'enable_kinematic_constraints': True,
        'enable_outlier_detection': True,
    }
    
    # 测试参数网格
    distance_values = [1800, 1900, 2000, 2100, 2200, 2300]
    speed_values = [800, 1000, 1200, 1500, 2000]
    
    print(f"\n📋 实验配置:")
    print(f"   基础配置: v3.4")
    print(f"   测试参数: update_distance × update_speed")
    print(f"   update_distance: {distance_values}")
    print(f"   update_speed: {speed_values}")
    print(f"   总实验数: {len(distance_values) * len(speed_values)}")
    print(f"   数据集: validation_set")
    
    # 运行网格搜索
    results = []
    total = len(distance_values) * len(speed_values)
    current = 0
    
    for dist, speed in itertools.product(distance_values, speed_values):
        current += 1
        print(f"\n进度: {current}/{total}")
        
        result = run_experiment(dist, speed, base_config, data_paths)
        if result:
            results.append(result)
    
    # 汇总结果
    print(f"\n{'='*80}")
    print(f"📊 实验结果汇总")
    print(f"{'='*80}")
    
    if not results:
        print("❌ 没有成功的实验")
        return
    
    results_df = pd.DataFrame(results)
    
    # 找最优组合
    best_result = results_df.loc[results_df['score'].idxmax()]
    baseline_score = results_df[
        (results_df['update_distance'] == 2000) & 
        (results_df['update_speed'] == 1000)
    ]['score'].values[0]
    
    results_df['improvement'] = results_df['score'] - baseline_score
    results_df['improvement_pct'] = (results_df['improvement'] / baseline_score) * 100
    
    # 按得分排序显示
    results_sorted = results_df.sort_values('score', ascending=False)
    print(f"\n前10名组合:")
    print(results_sorted.head(10)[['update_distance', 'update_speed', 'score', 'improvement_pct']].to_string(index=False))
    
    print(f"\n{'='*80}")
    print(f"🏆 最优结果")
    print(f"{'='*80}")
    print(f"   update_distance = {best_result['update_distance']}")
    print(f"   update_speed = {best_result['update_speed']}")
    print(f"   得分 = {best_result['score']:.4f}")
    print(f"\n   相比 baseline (2000, 1000):")
    print(f"   提升: {best_result['improvement']:+.4f} 分 ({best_result['improvement_pct']:+.2f}%)")
    
    # 保存结果
    output_file = data_paths['output_base'] / 'summary.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\n结果已保存到: {output_file}")


if __name__ == '__main__':
    main()

