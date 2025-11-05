"""
基线版本：不校准，直接用初赛20分方案
用于建立对比基准
"""
import subprocess
import sys
import os
import pandas as pd

def main():
    print("="*60)
    print("🚢 基线版本：初赛20分方案（不校准）")
    print("="*60)
    print("目标：建立基准，看复赛数据直接跑能得多少分")
    print("="*60)
    
    # 检查数据文件
    if not os.path.exists('radar_detection.csv'):
        print("❌ 错误: 找不到 radar_detection.csv（复赛数据）")
        sys.exit(1)
    
    # 步骤1: 预处理
    print("\n" + "="*60)
    print("🔧 步骤1: 数据预处理")
    print("="*60)
    result = subprocess.run([sys.executable, '20_ans/preprocess.py'], 
                          capture_output=False)
    
    if result.returncode != 0:
        print("❌ 预处理失败")
        sys.exit(1)
    
    # 步骤2: 轨迹生成
    print("\n" + "="*60)
    print("🚢 步骤2: 轨迹生成")
    print("="*60)
    result = subprocess.run([sys.executable, '20_ans/simple_interpolation-bak.py'], 
                          capture_output=False)
    
    if result.returncode != 0:
        print("❌ 轨迹生成失败")
        sys.exit(1)
    
    # 检查结果
    if os.path.exists('results.csv'):
        df = pd.read_csv('results.csv')
        print("\n" + "="*60)
        print("🎉 基线版本完成!")
        print("="*60)
        print(f"\n📊 统计:")
        print(f"  - 文件: results.csv")
        print(f"  - 轨迹点数: {len(df)}")
        print(f"  - 船舶数量: {df['ID'].nunique()}")
        print(f"  - 时间范围: {df['Time'].min()} ~ {df['Time'].max()}")
        
        print(f"\n✅ 可以提交 results.csv 到竞赛平台了！")
        print(f"\n💡 这是基线版本（不校准），记录得分后再尝试优化版本")
    else:
        print("\n❌ 错误: results.csv 未生成")
        sys.exit(1)

if __name__ == '__main__':
    main()

