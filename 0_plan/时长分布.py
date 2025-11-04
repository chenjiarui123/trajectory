"""
统计训练集轨迹的时间跨度
目的：优化外推时长参数
"""

import pandas as pd
import numpy as np
import glob
from pathlib import Path

def main():
    print("=" * 70)
    print("训练集轨迹时间跨度统计")
    print("=" * 70)
    
    gt_files = glob.glob('official/train/ground_truth/*.csv')
    print(f"\n找到 {len(gt_files)} 个轨迹文件")
    
    durations = []  # 时间跨度（分钟）
    num_points_list = []  # 轨迹点数
    
    print("\n正在分析...")
    for idx, gt_file in enumerate(gt_files):
        if (idx + 1) % 500 == 0:
            print(f"  进度: {idx+1}/{len(gt_files)}")
        
        try:
            df = pd.read_csv(gt_file)
            df['Time'] = pd.to_datetime(df['Time'])
            
            if len(df) < 2:
                continue
            
            # 计算时间跨度
            duration_minutes = (df['Time'].max() - df['Time'].min()).total_seconds() / 60
            durations.append(duration_minutes)
            num_points_list.append(len(df))
            
        except Exception as e:
            continue
    
    durations = np.array(durations)
    num_points = np.array(num_points_list)
    
    print("\n" + "=" * 70)
    print("统计结果")
    print("=" * 70)
    
    print("\n【轨迹时间跨度】(分钟)")
    print(f"  最小值:     {np.min(durations):.1f} 分钟")
    print(f"  5%分位:     {np.percentile(durations, 5):.1f} 分钟")
    print(f"  25%分位:    {np.percentile(durations, 25):.1f} 分钟")
    print(f"  中位数:     {np.median(durations):.1f} 分钟")
    print(f"  平均值:     {np.mean(durations):.1f} 分钟")
    print(f"  75%分位:    {np.percentile(durations, 75):.1f} 分钟")
    print(f"  95%分位:    {np.percentile(durations, 95):.1f} 分钟")
    print(f"  99%分位:    {np.percentile(durations, 99):.1f} 分钟")
    print(f"  最大值:     {np.max(durations):.1f} 分钟")
    
    print("\n【轨迹点数】")
    print(f"  最小值:     {np.min(num_points):.0f} 个")
    print(f"  中位数:     {np.median(num_points):.0f} 个")
    print(f"  平均值:     {np.mean(num_points):.1f} 个")
    print(f"  最大值:     {np.max(num_points):.0f} 个")
    
    print("\n【时间跨度分布】")
    ranges = [
        (0, 100, "< 100分钟"),
        (100, 200, "100-200分钟"),
        (200, 300, "200-300分钟"),
        (300, 400, "300-400分钟"),
        (400, 500, "400-500分钟"),
        (500, float('inf'), "> 500分钟")
    ]
    
    for min_val, max_val, label in ranges:
        count = np.sum((durations >= min_val) & (durations < max_val))
        pct = count / len(durations) * 100
        print(f"  {label:15s}: {count:4d} 艘 ({pct:5.1f}%)")
    
    print("\n" + "=" * 70)
    print("建议的外推时长参数")
    print("=" * 70)
    
    # 建议：覆盖90%的轨迹
    recommended = np.percentile(durations, 90)
    print(f"\n覆盖90%轨迹: {recommended:.0f} 分钟")
    print(f"覆盖95%轨迹: {np.percentile(durations, 95):.0f} 分钟")
    print(f"覆盖99%轨迹: {np.percentile(durations, 99):.0f} 分钟")
    
    print(f"\n当前使用: 200 分钟")
    coverage_200 = np.sum(durations <= 200) / len(durations) * 100
    print(f"200分钟覆盖率: {coverage_200:.1f}%")
    
    print(f"\n推荐参数:")
    if recommended <= 200:
        print(f"  保持200分钟即可（已覆盖90%+）")
    else:
        print(f"  建议增加到 {recommended:.0f} 分钟")
        print(f"  或者使用动态外推（根据单ESM范围调整）")


if __name__ == '__main__':
    main()

