"""
从训练集创建验证数据集
目标：构造一个和测试集结构相同的验证集，用于评测算法

策略：
1. 分层抽样：按交叉点数量分层（低质量<2，中等2-4，高质量>=5）
2. 合并为单个radar_detection.csv（类似测试集）
3. 保留对应的ground_truth作为答案
"""

import pandas as pd
import numpy as np
import glob
from pathlib import Path
import shutil

def analyze_crossing_points(esm_file):
    """快速分析交叉点数量"""
    try:
        df = pd.read_csv(esm_file)
        df['Time'] = pd.to_datetime(df['Time'])
        
        # ESM数据
        esm_data = df[df['Distance'] == -100]
        
        # 按时间分组，统计双ESM时刻数
        crossing_count = 0
        for time, group in esm_data.groupby('Time'):
            if len(group) >= 2:
                crossing_count += 1
        
        return crossing_count
    except:
        return 0


def main():
    print("=" * 70)
    print("从训练集创建验证数据集（分层抽样）")
    print("=" * 70)
    
    # 1. 扫描所有训练集船舶
    print("\n步骤1: 扫描训练集船舶...")
    train_dir = Path('official/train')
    radar_files = list(train_dir.glob('radar_detection/*.csv'))
    
    print(f"  找到 {len(radar_files)} 艘船的雷达数据")
    
    # 2. 统计每艘船的交叉点数量
    print("\n步骤2: 统计交叉点分布...")
    boat_stats = []
    
    for idx, radar_file in enumerate(radar_files):
        if (idx + 1) % 500 == 0:
            print(f"  进度: {idx+1}/{len(radar_files)}")
        
        boat_id = int(radar_file.stem)
        crossing_count = analyze_crossing_points(radar_file)
        
        boat_stats.append({
            'boat_id': boat_id,
            'crossing_points': crossing_count
        })
    
    stats_df = pd.DataFrame(boat_stats)
    
    # 3. 分层
    print("\n步骤3: 分层统计...")
    stats_df['quality'] = stats_df['crossing_points'].apply(
        lambda x: 'low' if x < 2 else ('medium' if x < 5 else 'high')
    )
    
    print("\n训练集分布:")
    for quality in ['low', 'medium', 'high']:
        count = len(stats_df[stats_df['quality'] == quality])
        pct = count / len(stats_df) * 100
        print(f"  {quality:8s}: {count:4d} 艘 ({pct:5.1f}%)")
    
    print(f"\n交叉点统计:")
    print(f"  平均: {stats_df['crossing_points'].mean():.2f}")
    print(f"  中位数: {stats_df['crossing_points'].median():.0f}")
    print(f"  最大: {stats_df['crossing_points'].max()}")
    print(f"  最小: {stats_df['crossing_points'].min()}")
    
    # 4. 分层抽样100艘 - 参考复赛分布
    print("\n步骤4: 分层抽样100艘船（参考复赛分布）...")

    # 复赛分布目标比例
    # low: 0%, medium: 8.6%, high: 91.4%
    total_sample = 100
    target_distribution = {
        'low': 0.0,      # 0艘
        'medium': 0.086, # 约9艘
        'high': 0.914    # 约91艘
    }

    print(f"\n目标分布（参考复赛）:")
    print(f"  low    (交叉点 <2):  {int(total_sample * target_distribution['low']):3d} 艘 ({target_distribution['low']*100:5.1f}%)")
    print(f"  medium (交叉点 2-4): {int(total_sample * target_distribution['medium']):3d} 艘 ({target_distribution['medium']*100:5.1f}%)")
    print(f"  high   (交叉点 >=5): {int(total_sample * target_distribution['high']):3d} 艘 ({target_distribution['high']*100:5.1f}%)")

    sampled_boats = []

    for quality in ['low', 'medium', 'high']:
        quality_df = stats_df[stats_df['quality'] == quality]

        # 使用复赛目标比例
        n_sample = int(total_sample * target_distribution[quality])

        # 随机抽样
        if n_sample > 0 and len(quality_df) > 0:
            sample = quality_df.sample(n=min(n_sample, len(quality_df)), random_state=42)
            sampled_boats.append(sample)
            print(f"  {quality:8s}: 抽样 {len(sample):2d} 艘 (目标 {n_sample:2d} 艘)")
        elif n_sample == 0:
            print(f"  {quality:8s}: 抽样  0 艘 (目标  0 艘) - 跳过")
        else:
            print(f"  {quality:8s}: 可用 {len(quality_df):2d} 艘，不足目标 {n_sample:2d} 艘")

    sampled_df = pd.concat(sampled_boats) if sampled_boats else pd.DataFrame()

    # 如果不足100艘，优先从high补充，其次medium
    if len(sampled_df) < total_sample:
        remaining = total_sample - len(sampled_df)
        print(f"\n  当前抽样 {len(sampled_df)} 艘，还需补充 {remaining} 艘")

        # 优先从high补充
        high_df = stats_df[
            (stats_df['quality'] == 'high') &
            (~stats_df['boat_id'].isin(sampled_df['boat_id']))
        ]
        if len(high_df) > 0:
            extra_high = high_df.sample(n=min(remaining, len(high_df)), random_state=43)
            sampled_df = pd.concat([sampled_df, extra_high])
            remaining -= len(extra_high)
            print(f"  从 high 补充: {len(extra_high)} 艘")

        # 如果还不够，从medium补充
        if remaining > 0:
            medium_df = stats_df[
                (stats_df['quality'] == 'medium') &
                (~stats_df['boat_id'].isin(sampled_df['boat_id']))
            ]
            if len(medium_df) > 0:
                extra_medium = medium_df.sample(n=min(remaining, len(medium_df)), random_state=44)
                sampled_df = pd.concat([sampled_df, extra_medium])
                print(f"  从 medium 补充: {len(extra_medium)} 艘")
    
    sampled_boat_ids = sampled_df['boat_id'].tolist()

    print(f"\n最终抽样: {len(sampled_boat_ids)} 艘船")

    # 验证最终分布
    print(f"\n最终分布验证:")
    for quality in ['low', 'medium', 'high']:
        count = len(sampled_df[sampled_df['quality'] == quality])
        pct = count / len(sampled_df) * 100 if len(sampled_df) > 0 else 0
        target_pct = target_distribution[quality] * 100
        diff = pct - target_pct
        status = "✓" if abs(diff) < 5 else "⚠"
        print(f"  {quality:8s}: {count:3d} 艘 ({pct:5.1f}%) - 目标 {target_pct:5.1f}% {status}")

    print(f"\n抽样后交叉点统计:")
    print(f"  平均: {sampled_df['crossing_points'].mean():.2f}")
    print(f"  中位数: {sampled_df['crossing_points'].median():.0f}")
    print(f"  最大: {sampled_df['crossing_points'].max()}")
    print(f"  最小: {sampled_df['crossing_points'].min()}")
    
    # 5. 创建验证集目录
    print("\n步骤5: 创建验证集...")
    val_dir = Path('validation_set')
    val_dir.mkdir(exist_ok=True)
    
    # 5.1 合并雷达数据（模拟测试集：ESM保留ID，普通雷达隐藏ID）
    print("  合并雷达数据...")
    all_radar_data = []
    
    for idx, boat_id in enumerate(sampled_boat_ids):
        if (idx + 1) % 20 == 0:
            print(f"    进度: {idx+1}/{len(sampled_boat_ids)}")
        
        radar_file = train_dir / f'radar_detection/{boat_id}.csv'
        if radar_file.exists():
            df = pd.read_csv(radar_file)
            
            # 模拟测试集规则：
            # - ESM数据（Distance == -100）：保留真实BoatID
            # - 普通雷达数据（Distance != -100）：BoatID设为-1（隐藏）
            df.loc[df['Distance'] != -100, 'BoatID'] = -1
            
            all_radar_data.append(df)
    
    merged_radar = pd.concat(all_radar_data, ignore_index=True)
    merged_radar = merged_radar.sort_values('Time').reset_index(drop=True)
    
    radar_output = val_dir / 'radar_detection.csv'
    merged_radar.to_csv(radar_output, index=False)
    print(f"  [完成] 雷达数据已保存: {radar_output}")
    print(f"     总记录数: {len(merged_radar)}")
    
    # 统计数据类型
    esm_count = len(merged_radar[merged_radar['Distance'] == -100])
    radar_count = len(merged_radar[merged_radar['Distance'] != -100])
    print(f"     ESM数据: {esm_count} 条 (BoatID保留)")
    print(f"     雷达数据: {radar_count} 条 (BoatID=-1，算法需自行判断归属)")
    
    # 5.2 复制ground_truth
    print("\n  复制ground truth...")
    gt_dir = val_dir / 'ground_truth'
    gt_dir.mkdir(exist_ok=True)
    
    for idx, boat_id in enumerate(sampled_boat_ids):
        if (idx + 1) % 20 == 0:
            print(f"    进度: {idx+1}/{len(sampled_boat_ids)}")
        
        gt_file = train_dir / f'ground_truth/{boat_id}.csv'
        if gt_file.exists():
            shutil.copy(gt_file, gt_dir / f'{boat_id}.csv')
    
    print(f"  [完成] Ground truth已复制到: {gt_dir}")
    
    print("\n" + "=" * 70)
    print("验证集创建完成！")
    print("=" * 70)
    print(f"\n验证集位置: {val_dir.absolute()}")
    print(f"船舶数量: {len(sampled_boat_ids)}")
    print(f"总记录数: {len(merged_radar)}")
    print(f"\n使用方法:")
    print(f"  1. 将算法中的 'official/test/radar_detection.csv'")
    print(f"     替换为 'validation_set/radar_detection.csv'")
    print(f"  2. 生成预测结果")
    print(f"  3. 与 'validation_set/ground_truth/*.csv' 对比评分")


if __name__ == '__main__':
    main()

