"""
快速调优脚本 - 只跑插值+匹配层，复用预处理结果

工程化设计：
1. 预处理结果缓存在 our_ans/preprocessed_cache/<dataset_name>/
2. 只需要跑一次预处理，后续所有调参实验都复用
3. 单次实验时间从几分钟降到几十秒

用法：
    # 第一次：生成预处理缓存
    python fast_eval_v3_4.py --prepare validation
    
    # 快速调参（只跑插值+匹配）
    python fast_eval_v3_4.py --dataset validation --update_distance 2500
    python fast_eval_v3_4.py --dataset validation --update_distance 2500 --update_angle 70 --iteration_rounds 7
    
    # 批量测试
    python fast_eval_v3_4.py --dataset validation --grid update_distance=2000,2500,3000
"""

import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
import pandas as pd

sys.path.append('.')

from common.ablation_wrapper import preprocess_with_params, interpolation_with_params
from common.评测脚本 import evaluate


# v3.4 的基准配置
BASELINE_CONFIG = {
    # 预处理参数
    'safe_distance': 200,
    'esm_angle_threshold': 0.065,
    'esm_speed_threshold': 1000,
    'esm_check_distance': 200000,
    'enable_deduplication': False,
    'enable_esm_merge': True,
    
    # 插值+匹配参数（这些是快速调优的目标）
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


def get_dataset_paths(dataset_name):
    """获取数据集路径"""
    if dataset_name == 'validation':
        return {
            'radar_file': 'validation_set/radar_detection.csv',
            'sensor_file': 'official/20_ans/sensors.txt',
            'gt_dir': 'validation_set',
        }
    elif dataset_name == 'final':
        return {
            'radar_file': 'official/radar_detection.csv',
            'sensor_file': 'official/20_ans/sensors.txt',
            'gt_dir': None,
        }
    else:
        raise ValueError(f"未知数据集: {dataset_name}")


def prepare_cache(dataset_name, config=None):
    """
    生成预处理缓存

    这一步比较慢（几十秒到几分钟），但只需要跑一次
    """
    if config is None:
        config = BASELINE_CONFIG

    paths = get_dataset_paths(dataset_name)
    cache_dir = Path('preprocessed_cache') / dataset_name
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print(f"🔧 生成预处理缓存: {dataset_name}")
    print("="*80)
    print(f"缓存目录: {cache_dir}")
    print("\n这一步比较慢，但只需要跑一次...")
    
    # 运行预处理
    preprocess_with_params(
        paths['radar_file'],
        paths['sensor_file'],
        str(cache_dir),
        safe_distance=config['safe_distance'],
        esm_angle_threshold=config['esm_angle_threshold'],
        esm_speed_threshold=config['esm_speed_threshold'],
        esm_check_distance=config['esm_check_distance'],
        enable_deduplication=config['enable_deduplication'],
        enable_esm_merge=config['enable_esm_merge']
    )
    
    # 保存配置信息
    cache_info = {
        'dataset': dataset_name,
        'created_at': datetime.now().isoformat(),
        'preprocess_config': {
            'safe_distance': config['safe_distance'],
            'esm_angle_threshold': config['esm_angle_threshold'],
            'esm_speed_threshold': config['esm_speed_threshold'],
            'esm_check_distance': config['esm_check_distance'],
            'enable_deduplication': config['enable_deduplication'],
            'enable_esm_merge': config['enable_esm_merge'],
        }
    }
    
    with open(cache_dir / 'cache_info.json', 'w', encoding='utf-8') as f:
        json.dump(cache_info, f, indent=2, ensure_ascii=False)
    
    print("\n✅ 预处理缓存生成完成！")
    print(f"\n缓存文件:")
    print(f"  - {cache_dir / 'located_ESM_points_updated.csv'}")
    print(f"  - {cache_dir / 'deduplacate_radar_points_updated.csv'}")
    print(f"\n现在可以快速调参了！")
    
    return cache_dir


def fast_eval(dataset_name, config_override, description=""):
    """
    快速评估 - 只跑插值+匹配层（串行版本）

    Args:
        dataset_name: 数据集名称
        config_override: 配置覆盖
        description: 实验描述

    这一步很快（几十秒），可以反复调参
    """
    # 检查缓存是否存在
    cache_dir = Path('preprocessed_cache') / dataset_name
    located_file = cache_dir / 'located_ESM_points_updated.csv'
    radar_file = cache_dir / 'deduplacate_radar_points_updated.csv'
    
    if not located_file.exists() or not radar_file.exists():
        print(f"❌ 预处理缓存不存在: {cache_dir}")
        print(f"\n请先运行: python fast_eval_v3_4.py --prepare {dataset_name}")
        return None, None
    
    # 合并配置
    config = BASELINE_CONFIG.copy()
    config.update(config_override)
    
    # 创建输出目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path('fast_tune_results') / dataset_name / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print(f"⚡ 快速评估: {description or timestamp}")
    print("="*80)
    
    # 显示修改的参数
    if config_override:
        print("\n修改的参数:")
        for key, value in config_override.items():
            old = BASELINE_CONFIG.get(key, 'N/A')
            print(f"  {key}: {old} → {value}")
    
    print(f"\n使用预处理缓存: {cache_dir}")
    print("使用串行模式...")

    try:
        # 串行插值
        interpolation_with_params(
            str(located_file),
            str(radar_file),
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
            enable_outlier_detection=config.get('enable_outlier_detection', False)
        )

        # 评测
        score = None
        num_points = 0
        paths = get_dataset_paths(dataset_name)

        result_file = output_dir / 'results.csv'
        if result_file.exists():
            results_df = pd.read_csv(result_file)
            num_points = len(results_df)

            if paths['gt_dir']:
                print("\n评测中...")
                score, _ = evaluate(
                    str(result_file),
                    paths['gt_dir'],
                    use_preliminary=True
                )

        # 保存实验记录
        result = {
            'timestamp': timestamp,
            'description': description,
            'score': score,
            'num_points': num_points,
            'output_dir': str(output_dir),
        }

        # 添加配置参数
        for key, value in config.items():
            result[f'config_{key}'] = value

        # 保存到日志
        log_file = Path('fast_tune_results') / dataset_name / 'experiments.csv'
        df = pd.DataFrame([result])
        if log_file.exists():
            existing = pd.read_csv(log_file)
            df = pd.concat([existing, df], ignore_index=True)
        df.to_csv(log_file, index=False)

        # 显示结果
        print("\n" + "="*80)
        print("✅ 完成！")
        print("="*80)
        if score is not None:
            print(f"得分: {score:.2f}")
        else:
            print(f"得分: N/A (无ground truth)")
        print(f"轨迹点数: {num_points:,}")
        print(f"输出目录: {output_dir}")

        # 检查是否是最佳得分
        if score and log_file.exists():
            all_results = pd.read_csv(log_file)
            if 'score' in all_results.columns:
                best_score = all_results['score'].max()
                if score >= best_score:
                    print(f"\n🏆 新的最佳得分！")
                    # 保存最佳配置
                    best_config_file = Path('fast_tune_results') / dataset_name / 'best_config.json'
                    best = {
                        'score': score,
                        'config': config,
                        'timestamp': timestamp,
                        'description': description
                    }
                    with open(best_config_file, 'w', encoding='utf-8') as f:
                        json.dump(best, f, indent=2, ensure_ascii=False)

        return score, result

    except Exception as e:
        print(f"\n❌ 失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def grid_search(dataset_name, param_grid):
    """网格搜索"""
    import itertools

    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))

    print(f"\n开始网格搜索: {len(combinations)} 个组合")

    results = []
    for i, combo in enumerate(combinations, 1):
        config_override = dict(zip(keys, combo))
        desc = ', '.join([f"{k}={v}" for k, v in config_override.items()])

        print(f"\n{'='*80}")
        print(f"实验 {i}/{len(combinations)}")
        print("="*80)

        score, result = fast_eval(dataset_name, config_override, desc)
        if score:
            results.append((score, config_override))

    # 显示结果
    if results:
        results.sort(reverse=True)
        print(f"\n{'='*80}")
        print("🏆 网格搜索结果 (按得分排序)")
        print("="*80)
        for i, (score, config) in enumerate(results[:5], 1):
            print(f"\n第{i}名: {score:.2f}")
            for k, v in config.items():
                print(f"  {k}: {v}")

    return results


def show_history(dataset_name, top_n=10):
    """显示历史实验"""
    log_file = Path('fast_tune_results') / dataset_name / 'experiments.csv'

    if not log_file.exists():
        print("还没有实验记录")
        return

    df = pd.read_csv(log_file)
    df = df.sort_values('score', ascending=False)

    print("="*80)
    print(f"📊 实验历史: {dataset_name}")
    print("="*80)
    print(f"\n总实验数: {len(df)}")
    if 'score' in df.columns and not df['score'].isna().all():
        print(f"最高得分: {df['score'].max():.2f}")
        print(f"最低得分: {df['score'].min():.2f}")
        print(f"平均得分: {df['score'].mean():.2f}")

    print(f"\n前{top_n}名:")
    cols = ['timestamp', 'description', 'score', 'num_points']
    available_cols = [c for c in cols if c in df.columns]
    print(df[available_cols].head(top_n).to_string(index=False))

    return df


def main():
    parser = argparse.ArgumentParser(description='v3.4 快速调优脚本')

    # 数据集选择
    parser.add_argument('--dataset', default='validation', choices=['validation', 'final'],
                       help='数据集名称')

    # 预处理缓存
    parser.add_argument('--prepare', metavar='DATASET', help='生成预处理缓存')

    # 单参数测试
    parser.add_argument('--update_distance', type=int, help='update_distance参数')
    parser.add_argument('--update_speed', type=int, help='update_speed参数')
    parser.add_argument('--update_angle', type=int, help='update_angle参数')
    parser.add_argument('--ransac_distance', type=int, help='ransac_distance参数')
    parser.add_argument('--ransac_min_points', type=int, help='ransac_min_points参数')
    parser.add_argument('--iteration_rounds', type=int, help='iteration_rounds参数')
    parser.add_argument('--interpolation_method', help='插值方法: linear/akima')

    # 网格搜索
    parser.add_argument('--grid', help='网格搜索，如: update_distance=2000,2500,3000')

    # 查看历史
    parser.add_argument('--history', action='store_true', help='查看实验历史')

    # 描述
    parser.add_argument('--desc', help='实验描述')

    args = parser.parse_args()

    # 生成预处理缓存
    if args.prepare:
        prepare_cache(args.prepare)
        return

    # 查看历史
    if args.history:
        show_history(args.dataset)
        return

    # 网格搜索
    if args.grid:
        param_grid = {}
        for item in args.grid.split():
            key, vals = item.split('=')
            vals = vals.split(',')
            # 尝试转换为数字
            try:
                vals = [int(v) for v in vals]
            except:
                try:
                    vals = [float(v) for v in vals]
                except:
                    pass
            param_grid[key] = vals

        grid_search(args.dataset, param_grid)
        return

    # 单次快速评估
    config_override = {}

    if args.update_distance is not None:
        config_override['update_distance'] = args.update_distance
    if args.update_speed is not None:
        config_override['update_speed'] = args.update_speed
    if args.update_angle is not None:
        config_override['update_angle'] = args.update_angle
    if args.ransac_distance is not None:
        config_override['ransac_distance'] = args.ransac_distance
    if args.ransac_min_points is not None:
        config_override['ransac_min_points'] = args.ransac_min_points
    if args.iteration_rounds is not None:
        config_override['iteration_rounds'] = args.iteration_rounds
    if args.interpolation_method is not None:
        config_override['interpolation_method'] = args.interpolation_method

    if not config_override:
        # 默认：运行baseline
        print("运行baseline配置...")
        config_override = {}

    fast_eval(args.dataset, config_override, args.desc or "")


if __name__ == '__main__':
    main()


