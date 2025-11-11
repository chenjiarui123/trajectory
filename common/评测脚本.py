"""
评测脚本 - 使用初赛评分公式
用于评测算法在验证集上的表现

评分公式：
S = (1/K) * Σ score(d(pₖ, qₖ))
score(d) = 100 * e^(-0.0046d)

其中：
- K：真值轨迹点的总个数（在有效侦测时间范围内）
- pₖ：第k时刻（分钟）的重建轨迹点
- qₖ：第k时刻（分钟）的真值轨迹点
- d(∙)：欧氏距离计算函数（平面坐标系，单位：米）
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# 添加项目根目录到路径，以便导入 official 模块
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from official.CoordinateConvert import lonlat_to_xy
import time as time_module

def calculate_distance_meters(lon1, lat1, lon2, lat2):
    """
    计算两点之间的欧氏距离（米）
    使用平面坐标系计算
    """
    xy1 = lonlat_to_xy(lon1, lat1)
    xy2 = lonlat_to_xy(lon2, lat2)
    distance = np.sqrt((xy1[0] - xy2[0])**2 + (xy1[1] - xy2[1])**2)
    return distance

def get_valid_time_range(radar_df, boat_id):
    """
    获取船舶的有效侦测时间范围
    从ESM数据中提取该船的最早和最晚时间
    """
    esm_data = radar_df[radar_df['Distance'] == -100]
    boat_esm = esm_data[esm_data['BoatID'] == boat_id]
    
    if len(boat_esm) == 0:
        return None, None
    
    min_time = boat_esm['Time'].min()
    max_time = boat_esm['Time'].max()
    
    return min_time, max_time

def evaluate_single_boat(pred_df, gt_df, radar_df, boat_id, decay_factor=0.0046):
    """
    评测单艘船的轨迹
    
    Args:
        pred_df: 预测结果DataFrame (ID, Time, LON, LAT)
        gt_df: 真值DataFrame (Time, ID, LON, LAT)
        radar_df: 雷达数据DataFrame（用于确定有效时间范围）
        boat_id: 船舶ID
        decay_factor: 评分公式衰减因子（初赛0.0046，复赛0.0013）
    
    Returns:
        score: 该船的得分
        valid_points: 有效评分点数
        total_points: 真值总点数
    """
    # 获取该船的有效时间范围
    min_time, max_time = get_valid_time_range(radar_df, boat_id)
    
    if min_time is None or max_time is None:
        return 0.0, 0, 0
    
    # 筛选该船的预测和真值
    pred_boat = pred_df[pred_df['ID'] == boat_id].copy()
    gt_boat = gt_df[gt_df['ID'] == boat_id].copy()
    
    if len(pred_boat) == 0 or len(gt_boat) == 0:
        return 0.0, 0, len(gt_boat)
    
    # 转换时间格式
    pred_boat['Time'] = pd.to_datetime(pred_boat['Time'])
    gt_boat['Time'] = pd.to_datetime(gt_boat['Time'])
    
    # 筛选有效时间范围内的真值点
    gt_valid = gt_boat[(gt_boat['Time'] >= min_time) & (gt_boat['Time'] <= max_time)].copy()
    
    if len(gt_valid) == 0:
        return 0.0, 0, len(gt_boat)
    
    # 将预测结果按时间建立索引（去重，保留第一个）
    pred_dict = {}
    for _, row in pred_boat.iterrows():
        t = row['Time']
        if t not in pred_dict:  # 只保留第一次出现的
            pred_dict[t] = (row['LON'], row['LAT'])
    
    # 计算得分
    total_score = 0.0
    valid_count = 0
    
    for _, gt_row in gt_valid.iterrows():
        t = gt_row['Time']
        gt_lon = gt_row['LON']
        gt_lat = gt_row['LAT']
        
        # 检查预测结果中是否有该时间点
        if t in pred_dict:
            pred_lon, pred_lat = pred_dict[t]
            
            # 计算距离（米）
            distance = calculate_distance_meters(gt_lon, gt_lat, pred_lon, pred_lat)
            
            # 计算得分
            score = 100 * np.exp(-decay_factor * distance)
            total_score += score
            valid_count += 1
    
    # 计算平均得分
    K = len(gt_valid)  # 有效时间范围内的真值点数
    if K == 0:
        return 0.0, 0, len(gt_boat)
    
    avg_score = total_score / K
    
    return avg_score, valid_count, len(gt_boat)

def evaluate(prediction_file, validation_dir='validation_set', use_preliminary=True):
    """
    评测函数
    
    Args:
        prediction_file: 预测结果文件路径
        validation_dir: 验证集目录
        use_preliminary: 是否使用初赛评分公式（True=初赛，False=复赛）
    
    Returns:
        final_score: 最终得分
        boat_scores: 每艘船的得分详情
    """
    print("=" * 70)
    print("开始评测")
    print("=" * 70)
    
    # 确定评分公式参数
    if use_preliminary:
        print("\n使用初赛评分公式: score(d) = 100 * e^(-0.0046d)")
        decay_factor = 0.0046
    else:
        print("\n使用复赛评分公式: score(d) = 100 * e^(-0.0013d)")
        decay_factor = 0.0013
    
    # 读取预测结果
    print(f"\n步骤1: 读取预测结果...")
    print(f"  文件: {prediction_file}")
    pred_df = pd.read_csv(prediction_file)
    pred_df['Time'] = pd.to_datetime(pred_df['Time'])
    print(f"  总记录数: {len(pred_df)}")
    print(f"  船舶数量: {pred_df['ID'].nunique()}")
    
    # 读取雷达数据（用于确定有效时间范围）
    print(f"\n步骤2: 读取雷达数据...")
    radar_file = Path(validation_dir) / 'radar_detection.csv'
    radar_df = pd.read_csv(radar_file)
    radar_df['Time'] = pd.to_datetime(radar_df['Time'])
    print(f"  文件: {radar_file}")
    print(f"  总记录数: {len(radar_df)}")
    
    # 读取所有ground truth文件
    print(f"\n步骤3: 读取ground truth...")
    gt_dir = Path(validation_dir) / 'ground_truth'
    gt_files = list(gt_dir.glob('*.csv'))
    print(f"  找到 {len(gt_files)} 个ground truth文件")
    
    # 合并所有ground truth
    gt_list = []
    for gt_file in gt_files:
        df = pd.read_csv(gt_file)
        gt_list.append(df)
    
    if len(gt_list) == 0:
        print("  错误: 未找到ground truth文件！")
        return 0.0, {}
    
    gt_df = pd.concat(gt_list, ignore_index=True)
    gt_df['Time'] = pd.to_datetime(gt_df['Time'])
    print(f"  总记录数: {len(gt_df)}")
    print(f"  船舶数量: {gt_df['ID'].nunique()}")
    
    # 获取所有需要评测的船舶ID
    boat_ids = sorted(gt_df['ID'].unique())
    print(f"\n步骤4: 开始评测 {len(boat_ids)} 艘船...")
    
    # 评测每艘船
    boat_scores = {}
    total_valid_points = 0
    total_gt_points = 0
    
    start_time = time_module.time()
    
    for idx, boat_id in enumerate(boat_ids):
        if (idx + 1) % 20 == 0:
            print(f"  进度: {idx+1}/{len(boat_ids)}")
        
        score, valid_points, gt_points = evaluate_single_boat(
            pred_df, gt_df, radar_df, boat_id, decay_factor
        )
        
        boat_scores[boat_id] = {
            'score': score,
            'valid_points': valid_points,
            'total_gt_points': gt_points
        }
        
        total_valid_points += valid_points
        total_gt_points += gt_points
    
    elapsed_time = time_module.time() - start_time
    
    # 计算最终得分（所有船舶的平均得分）
    if len(boat_scores) > 0:
        final_score = np.mean([boat_scores[bid]['score'] for bid in boat_scores])
    else:
        final_score = 0.0
    
    # 输出结果
    print("\n" + "=" * 70)
    print("评测结果")
    print("=" * 70)
    print(f"\n最终得分: {final_score:.6f}")
    print(f"评测船舶数: {len(boat_scores)}")
    print(f"有效评分点数: {total_valid_points}")
    print(f"真值总点数: {total_gt_points}")
    print(f"覆盖率: {total_valid_points/total_gt_points*100:.2f}%")
    print(f"处理时间: {elapsed_time:.2f} 秒")
    
    # 统计信息
    scores_list = [boat_scores[bid]['score'] for bid in boat_scores]
    print(f"\n得分统计:")
    print(f"  平均: {np.mean(scores_list):.6f}")
    print(f"  中位数: {np.median(scores_list):.6f}")
    print(f"  最大: {np.max(scores_list):.6f}")
    print(f"  最小: {np.min(scores_list):.6f}")
    print(f"  标准差: {np.std(scores_list):.6f}")
    
    # 显示得分分布
    print(f"\n得分分布:")
    excellent = sum(1 for s in scores_list if s >= 90)
    good = sum(1 for s in scores_list if 70 <= s < 90)
    fair = sum(1 for s in scores_list if 50 <= s < 70)
    poor = sum(1 for s in scores_list if s < 50)
    print(f"  优秀 (≥90): {excellent} 艘 ({excellent/len(scores_list)*100:.1f}%)")
    print(f"  良好 (70-90): {good} 艘 ({good/len(scores_list)*100:.1f}%)")
    print(f"  一般 (50-70): {fair} 艘 ({fair/len(scores_list)*100:.1f}%)")
    print(f"  较差 (<50): {poor} 艘 ({poor/len(scores_list)*100:.1f}%)")
    
    return final_score, boat_scores

def main():
    """主函数"""
    import sys
    
    # 默认参数（相对于项目根目录）
    prediction_file = 'results.csv'
    validation_dir = 'validation_set'
    use_preliminary = True  # 使用初赛评分公式
    
    # 从命令行参数读取
    if len(sys.argv) > 1:
        prediction_file = sys.argv[1]
    if len(sys.argv) > 2:
        validation_dir = sys.argv[2]
    if len(sys.argv) > 3:
        use_preliminary = sys.argv[3].lower() in ['true', '1', 'yes', 'preliminary']
    
    # 如果文件不在当前目录，尝试从项目根目录查找
    project_root = Path(__file__).parent.parent
    if not Path(prediction_file).exists():
        potential_file = project_root / prediction_file
        if potential_file.exists():
            prediction_file = str(potential_file)
    
    if not Path(validation_dir).exists():
        potential_dir = project_root / validation_dir
        if potential_dir.exists():
            validation_dir = str(potential_dir)
    
    # 检查文件是否存在
    if not Path(prediction_file).exists():
        print(f"错误: 预测结果文件不存在: {prediction_file}")
        print(f"\n使用方法:")
        print(f"  python 评测脚本.py [预测文件] [验证集目录] [是否初赛评分]")
        print(f"\n示例:")
        print(f"  python 评测脚本.py submission_17_no_stepping.csv validation_set True")
        return
    
    # 运行评测
    final_score, boat_scores = evaluate(prediction_file, validation_dir, use_preliminary)
    
    # 保存详细结果
    output_file = 'evaluation_results.csv'
    results_list = []
    for boat_id, info in boat_scores.items():
        results_list.append({
            'BoatID': boat_id,
            'Score': info['score'],
            'ValidPoints': info['valid_points'],
            'TotalGTPoints': info['total_gt_points']
        })
    
    results_df = pd.DataFrame(results_list)
    results_df = results_df.sort_values('BoatID')
    results_df.to_csv(output_file, index=False)
    print(f"\n详细结果已保存到: {output_file}")

if __name__ == '__main__':
    main()

