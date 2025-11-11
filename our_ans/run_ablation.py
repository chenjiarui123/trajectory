"""
消融实验运行脚本 - 简单直接版本
针对 20_ans 方法进行参数消融实验
"""

import sys
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import os
from tqdm import tqdm

# 添加路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from common.评测脚本 import evaluate
from our_ans.ablation_wrapper import preprocess_with_params, interpolation_with_params

# 禁用所有子进程的tqdm输出
os.environ['TQDM_DISABLE'] = '1'


def run_single_experiment(exp_name, params, pbar=None):
    """
    运行单次消融实验

    Args:
        exp_name: 实验名称
        params: 参数字典
        pbar: tqdm进度条对象

    Returns:
        result: 包含得分和参数的字典
    """
    # 创建实验目录
    exp_dir = project_root / 'our_ans' / 'ablation_results'
    exp_dir.mkdir(exist_ok=True)

    exp_subdir = exp_dir / exp_name
    exp_subdir.mkdir(exist_ok=True)
    output_dir = exp_subdir / 'processed_data'

    # 保存配置
    config_file = exp_subdir / 'config.json'
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(params, f, indent=2, ensure_ascii=False)

    try:
        # 步骤1: 预处理
        if pbar:
            pbar.set_description(f"{exp_name} - 预处理")

        sensor_coords_file = project_root / 'our_ans' / '20_ans' / 'sensors.txt'
        radar_file = project_root / 'validation_set' / 'radar_detection.csv'

        # 重定向输出到null
        import io
        import contextlib

        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            preprocess_with_params(
                str(radar_file),
                str(sensor_coords_file),
                str(output_dir),
                safe_distance=params.get('safe_distance', 200),
                esm_angle_threshold=params.get('esm_angle_threshold', 0.001),
                esm_speed_threshold=params.get('esm_speed_threshold', 1000),
                esm_check_distance=params.get('esm_check_distance', 200000),
                enable_deduplication=params.get('enable_deduplication', True),
                enable_esm_merge=params.get('enable_esm_merge', True)
            )

        # 步骤2: 插值
        if pbar:
            pbar.set_description(f"{exp_name} - 插值")

        located_ESM_file = output_dir / 'located_ESM_points_updated.csv'
        radar_points_file = output_dir / 'deduplacate_radar_points_updated.csv'

        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            interpolation_with_params(
                str(located_ESM_file),
                str(radar_points_file),
                str(output_dir),
                update_distance=params.get('update_distance', 600),
                update_speed=params.get('update_speed', 800),
                update_angle=params.get('update_angle', 90),
                ransac_distance=params.get('ransac_distance', 500),
                ransac_min_points=params.get('ransac_min_points', 5),
                iteration_rounds=params.get('iteration_rounds', 3),
                enable_ransac=params.get('enable_ransac', True)
            )

        # 步骤3: 评测
        if pbar:
            pbar.set_description(f"{exp_name} - 评测")

        result_file = output_dir / 'results.csv'

        if not result_file.exists():
            score = 0.0
        else:
            with contextlib.redirect_stdout(io.StringIO()):
                score, _ = evaluate(
                    str(result_file),
                    str(project_root / 'validation_set'),
                    use_preliminary=True
                )

        # 记录结果
        result = {
            'experiment': exp_name,
            'score': score,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            **params
        }

        # 保存单次结果
        result_json = exp_subdir / 'result.json'
        with open(result_json, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        if pbar:
            pbar.set_postfix({'得分': f'{score:.2f}'})

        return result

    except Exception as e:
        if pbar:
            pbar.set_postfix({'状态': '失败'})

        result = {
            'experiment': exp_name,
            'score': 0.0,
            'error': str(e),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            **params
        }
        return result


def run_optimal_config():
    """
    使用最优配置处理官方数据集
    基于消融实验结果，应用最优参数配置
    """
    print("\n" + "="*80)
    print("应用最优配置到官方数据集")
    print("="*80)

    # 最优配置 (基于消融实验结果)
    optimal_config = {
        # 预处理参数
        'safe_distance': 200,                    # 去重距离 (影响不大)
        'esm_angle_threshold': 0.065,            # ⭐⭐⭐ 最关键！0.060-0.070最优
        'esm_speed_threshold': 1000,             # 速度阈值 (影响不大)
        'esm_check_distance': 200000,            # ⭐ 保持200km
        'enable_deduplication': False,           # ⭐ 关闭去重 (+0.3%)
        'enable_esm_merge': True,                # ⭐ 必须开启
        # 插值参数
        'update_distance': 600,
        'update_speed': 800,
        'update_angle': 90,
        'ransac_distance': 500,
        'ransac_min_points': 5,
        'iteration_rounds': 5,                   # ⭐ 增加到5轮 (+1.2%)
        'enable_ransac': True,                   # ⭐⭐⭐ 必须开启
    }

    print("\n最优配置参数:")
    print("-" * 80)
    print(f"  ESM角度阈值:     {optimal_config['esm_angle_threshold']:.3f}° (关键参数，提升27.7%)")
    print(f"  去重模块:        {'关闭' if not optimal_config['enable_deduplication'] else '开启'} (关闭提升0.3%)")
    print(f"  插值迭代次数:    {optimal_config['iteration_rounds']}轮 (提升1.2%)")
    print(f"  RANSAC模块:      {'开启' if optimal_config['enable_ransac'] else '关闭'} (必须开启)")
    print(f"  ESM合并模块:     {'开启' if optimal_config['enable_esm_merge'] else '关闭'} (必须开启)")
    print("-" * 80)
    print(f"\n预期得分提升: ~29% (从22.83到约29.5)")

    # 设置路径
    radar_file = project_root / 'official' / 'radar_detection.csv'
    sensor_coords_file = project_root / 'our_ans' / '20_ans' / 'sensors.txt'
    output_dir = project_root / 'our_ans' / 'optimal_results'
    output_dir.mkdir(exist_ok=True)

    print(f"\n输入文件: {radar_file}")
    print(f"输出目录: {output_dir}")

    # 保存配置
    config_file = output_dir / 'optimal_config.json'
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(optimal_config, f, indent=2, ensure_ascii=False)
    print(f"配置已保存: {config_file}")

    try:
        # 步骤1: 预处理
        print("\n" + "="*80)
        print("步骤 1/2: 预处理")
        print("="*80)

        import io
        import contextlib

        # 显示进度但隐藏详细输出
        print("正在处理...")
        print("  - ESM交叉定位")
        print("  - 雷达点去重 (已关闭)")
        print("  - ESM-雷达点匹配")

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
        print("步骤 2/2: 轨迹插值")
        print("="*80)

        located_ESM_file = output_dir / 'located_ESM_points_updated.csv'
        radar_points_file = output_dir / 'deduplacate_radar_points_updated.csv'

        print("正在处理...")
        print(f"  - 曲线拟合")
        print(f"  - RANSAC轨迹扩展")
        print(f"  - 迭代优化 ({optimal_config['iteration_rounds']}轮)")

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
            enable_ransac=optimal_config['enable_ransac']
        )

        print("✓ 插值完成")

        # 结果文件
        result_file = output_dir / 'results.csv'

        print("\n" + "="*80)
        print("处理完成！")
        print("="*80)
        print(f"\n结果文件: {result_file}")

        # 检查结果
        if result_file.exists():
            df = pd.read_csv(result_file)
            print(f"\n结果统计:")
            print(f"  总轨迹点数: {len(df):,}")
            print(f"  船只数量:   {df['ID'].nunique():,}")
            print(f"  时间范围:   {df['Time'].min()} - {df['Time'].max()}")

            # 显示前几行
            print(f"\n结果预览:")
            print(df.head(10).to_string(index=False))

        print("\n" + "="*80)
        print("✅ 最优配置应用成功！")
        print("="*80)

        return str(result_file)

    except Exception as e:
        print(f"\n❌ 处理失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """主函数 - 定义并运行所有实验"""

    print("="*80)
    print("20_ans 方法消融实验")
    print("="*80)

    # 基线配置
    baseline = {
        # 预处理参数
        'safe_distance': 200,
        'esm_angle_threshold': 0.001,
        'esm_speed_threshold': 1000,
        'esm_check_distance': 200000,
        'enable_deduplication': True,
        'enable_esm_merge': True,
        # 插值参数
        'update_distance': 600,
        'update_speed': 800,
        'update_angle': 90,
        'ransac_distance': 500,
        'ransac_min_points': 5,
        'iteration_rounds': 3,
        'enable_ransac': True,
    }

    # ========== ESM角度阈值细化实验 ==========
    # 目标: 找出ESM角度阈值的最优值
    # 已知: 0.005° 得分24.97，提升曲线还在加速
    # 策略: 在0.005°附近和更大范围探索上限

    print("="*80)
    print("ESM角度阈值细化实验")
    print("="*80)
    print("\n已知结果:")
    print("  0.001° (baseline): 22.83")
    print("  0.002°:            23.73 (+3.9%)")
    print("  0.005°:            24.97 (+9.4%) ⬆️ 提升加速")
    print("\n实验目标: 找出最优角度阈值上限\n")

    experiments = []

    # 基线对比
    experiments.append(('baseline_0.001', baseline))
    experiments.append(('angle_0.005', {**baseline, 'esm_angle_threshold': 0.005}))

    # 细化实验: 0.005°附近
    experiments.extend([
        ('angle_0.006', {**baseline, 'esm_angle_threshold': 0.006}),
        ('angle_0.007', {**baseline, 'esm_angle_threshold': 0.007}),
        ('angle_0.008', {**baseline, 'esm_angle_threshold': 0.008}),
        ('angle_0.009', {**baseline, 'esm_angle_threshold': 0.009}),
        ('angle_0.010', {**baseline, 'esm_angle_threshold': 0.010}),
    ])

    # 探索上限: 更大的角度
    experiments.extend([
        ('angle_0.012', {**baseline, 'esm_angle_threshold': 0.012}),
        ('angle_0.015', {**baseline, 'esm_angle_threshold': 0.015}),
        ('angle_0.020', {**baseline, 'esm_angle_threshold': 0.020}),
        ('angle_0.030', {**baseline, 'esm_angle_threshold': 0.030}),
    ])

    # 探索下限: 验证是否还有更小的最优点
    experiments.extend([
        ('angle_0.003', {**baseline, 'esm_angle_threshold': 0.003}),
        ('angle_0.004', {**baseline, 'esm_angle_threshold': 0.004}),
    ])

    print(f"\n共 {len(experiments)} 个实验配置\n")

    # 运行所有实验 - 使用tqdm进度条
    results = []
    with tqdm(experiments, desc="总进度", unit="实验") as pbar:
        for exp_name, params in pbar:
            result = run_single_experiment(exp_name, params, pbar)
            results.append(result)

    # 保存汇总结果
    print("\n" + "="*80)
    print("ESM角度阈值细化实验 - 结果汇总")
    print("="*80)

    df = pd.DataFrame(results)
    df = df.sort_values('score', ascending=False)

    exp_dir = project_root / 'our_ans' / 'ablation_results'
    summary_file = exp_dir / 'angle_threshold_summary.csv'
    df.to_csv(summary_file, index=False, encoding='utf-8-sig')

    # 获取baseline得分
    baseline_score = df[df['experiment'] == 'baseline_0.001']['score'].values[0]

    print("\n实验结果 (按得分排序):")
    print("="*80)
    print(f"{'排名':<6} {'实验名称':<20} {'角度阈值':<12} {'得分':<10} {'相对baseline':<15}")
    print("-" * 80)

    for idx, (_, row) in enumerate(df.iterrows(), 1):
        rank_icon = "🥇" if idx == 1 else "🥈" if idx == 2 else "🥉" if idx == 3 else f"{idx:2d}"
        angle = row['esm_angle_threshold']
        score = row['score']
        diff = score - baseline_score
        pct = (diff / baseline_score) * 100

        print(f"{rank_icon:<6} {row['experiment']:<20} {angle:8.4f}°    {score:7.2f}    {diff:+6.2f} ({pct:+5.1f}%)")

    print(f"\n详细结果已保存到: {summary_file}")

    # 找出最佳配置
    best = df.iloc[0]
    improvement = best['score'] - baseline_score
    improvement_pct = (improvement / baseline_score) * 100

    print("\n" + "="*80)
    print("🏆 最佳角度阈值")
    print("="*80)
    print(f"实验名称: {best['experiment']}")
    print(f"角度阈值: {best['esm_angle_threshold']:.4f}°")
    print(f"得分: {best['score']:.2f}")
    print(f"baseline得分: {baseline_score:.2f}")
    print(f"提升: {improvement:+.2f} ({improvement_pct:+.2f}%)")

    # 趋势分析
    print("\n" + "="*80)
    print("📈 角度阈值 vs 得分趋势")
    print("="*80)

    trend_df = df.sort_values('esm_angle_threshold')
    print(f"\n{'角度阈值':<12} {'得分':<10} {'可视化'}")
    print("-" * 60)
    for _, row in trend_df.iterrows():
        angle = row['esm_angle_threshold']
        score = row['score']
        bar_length = int((score - 14) / 0.2)  # 简单的条形图
        bar = "█" * bar_length
        print(f"{angle:8.4f}°    {score:7.2f}   {bar}")

    print("\n" + "="*80)


if __name__ == '__main__':
    import sys

    # 检查命令行参数
    if len(sys.argv) > 1 and sys.argv[1] == '--optimal':
        # 应用最优配置到官方数据集
        run_optimal_config()
    else:
        # 运行消融实验
        print("\n提示: 使用 'python run_ablation.py --optimal' 来应用最优配置到官方数据集\n")
        main()

