import pandas as pd
import numpy as np
from pathlib import Path
from official.sensor_config import SENSOR_INFO, ESM_SENSORS
from official.CoordinateConvert import lonlat_to_xy


class SimpleDataAnalyzer:
    """简化的数据分析器"""
    
    def __init__(self):
        # 准备ESM传感器位置的平面坐标
        self.sensor_xy_dict = {}
        for sid in ESM_SENSORS:
            info = SENSOR_INFO[sid]
            self.sensor_xy_dict[sid] = lonlat_to_xy(info['lon'], info['lat'])
    
    def count_crossings(self, df, boat_id=None):
        """统计交叉点数量"""
        # 筛选ESM数据
        esm_data = df[df['Distance'] == -100.0].copy()
        
        if boat_id is not None:
            esm_data = esm_data[esm_data['BoatID'] == boat_id]
        
        # 转换时间格式
        if not pd.api.types.is_datetime64_any_dtype(esm_data['Time']):
            esm_data['Time'] = pd.to_datetime(esm_data['Time'])
        
        # 按时间分组，统计双ESM交叉点
        grouped = esm_data.groupby('Time')
        crossing_count = 0
    
        for time, group in grouped:
            sensors = group['SensorID'].values
            valid_sensors = [s for s in sensors if s in ESM_SENSORS]
            if len(valid_sensors) >= 2:
                crossing_count += 1
        
        return crossing_count
    
    def analyze_train_data(self, train_dir='official/train/radar_detection'):
        """分析训练集数据"""
        print("\n" + "="*70)
        print("[分析训练集数据]")
        print("="*70)
        
        train_path = Path(train_dir)
        if not train_path.exists():
            print(f"[错误] 训练集目录不存在: {train_path}")
            return None
        
        boat_files = list(train_path.glob('*.csv'))
        print(f"发现 {len(boat_files)} 艘船的数据文件")
        
        all_stats = []
        
        for i, boat_file in enumerate(boat_files, 1):
            if i % 500 == 0:
                print(f"  进度: {i}/{len(boat_files)}")
            
            try:
                boat_id = int(boat_file.stem)
                df = pd.read_csv(boat_file)
                
                crossing_count = self.count_crossings(df, boat_id)
                total_2d = len(df[df['Distance'] > 0])
                
                all_stats.append({
                    'BoatID': boat_id,
                    'CrossingPoints': crossing_count,
                    'Total2DPoints': total_2d
                })
            except Exception as e:
                continue
        
        stats_df = pd.DataFrame(all_stats)
        
        # 打印统计摘要
        print(f"\n[训练集统计摘要]")
        print(f"  总船舶数: {len(stats_df)}")
        print(f"  平均交叉点: {stats_df['CrossingPoints'].mean():.2f}")
        print(f"  中位数交叉点: {stats_df['CrossingPoints'].median():.0f}")
        print(f"  最大交叉点: {stats_df['CrossingPoints'].max()}")
        print(f"  最小交叉点: {stats_df['CrossingPoints'].min()}")
        
        # 数据分层
        high = len(stats_df[stats_df['CrossingPoints'] >= 5])
        medium = len(stats_df[(stats_df['CrossingPoints'] >= 2) & (stats_df['CrossingPoints'] < 5)])
        low = len(stats_df[stats_df['CrossingPoints'] < 2])
        
        print(f"\n[数据质量分层]")
        print(f"  高质量(>=5): {high} 艘 ({high/len(stats_df)*100:.1f}%)")
        print(f"  中等(2-4):   {medium} 艘 ({medium/len(stats_df)*100:.1f}%)")
        print(f"  低质量(<2):  {low} 艘 ({low/len(stats_df)*100:.1f}%)")
        
        return stats_df
    
    def analyze_test_data(self, test_file='official/test/radar_detection.csv'):
        """分析测试集数据"""
        print("\n" + "="*70)
        print("[分析测试集数据]")
        print("="*70)
        
        test_path = Path(test_file)
        if not test_path.exists():
            print(f"[错误] 测试集文件不存在: {test_path}")
            return None
        
        df = pd.read_csv(test_file)
        df['Time'] = pd.to_datetime(df['Time'])
        
        esm_data = df[df['Distance'] == -100.0].copy()
        radar_2d_data = df[df['Distance'] > 0].copy()
        
        # 基础统计
        print(f"\n[基础统计]")
        print(f"  总记录数: {len(df)}")
        print(f"  ESM记录: {len(esm_data)} ({len(esm_data)/len(df)*100:.1f}%)")
        print(f"  2D雷达: {len(radar_2d_data)} ({len(radar_2d_data)/len(df)*100:.1f}%)")
        
        # ESM数据分析
        esm_with_id = esm_data[esm_data['BoatID'] != -1]
        esm_without_id = esm_data[esm_data['BoatID'] == -1]
        
        print(f"\n[ESM数据分析]")
        print(f"  有BoatID: {len(esm_with_id)} ({len(esm_with_id)/len(esm_data)*100:.1f}%)")
        print(f"  无BoatID: {len(esm_without_id)} ({len(esm_without_id)/len(esm_data)*100:.1f}%)")
        print(f"  涉及船舶: {esm_with_id['BoatID'].nunique()} 艘")
        
        # 分析每艘船的交叉点
        if len(esm_with_id) > 0:
            boat_stats = []
            for boat_id in esm_with_id['BoatID'].unique():
                boat_esm = esm_with_id[esm_with_id['BoatID'] == boat_id]
                grouped = boat_esm.groupby('Time')
                
                crossing_count = 0
                for time, group in grouped:
                    sensors = group['SensorID'].values
                    valid_sensors = [s for s in sensors if s in ESM_SENSORS]
                    if len(valid_sensors) >= 2:
                        crossing_count += 1
                
                boat_stats.append({
                    'BoatID': boat_id,
                    'CrossingPoints': crossing_count
                })
            
            boat_stats_df = pd.DataFrame(boat_stats)
            
            print(f"\n[交叉点统计]")
            print(f"  总交叉点: {boat_stats_df['CrossingPoints'].sum()}")
            print(f"  平均每艘船: {boat_stats_df['CrossingPoints'].mean():.2f}")
            print(f"  中位数: {boat_stats_df['CrossingPoints'].median():.0f}")
            print(f"  最大: {boat_stats_df['CrossingPoints'].max()}")
            print(f"  最小: {boat_stats_df['CrossingPoints'].min()}")
            
            # 数据质量分层
            high = len(boat_stats_df[boat_stats_df['CrossingPoints'] >= 5])
            medium = len(boat_stats_df[(boat_stats_df['CrossingPoints'] >= 2) & (boat_stats_df['CrossingPoints'] < 5)])
            low = len(boat_stats_df[boat_stats_df['CrossingPoints'] < 2])
            
            print(f"\n[数据质量分层]")
            print(f"  高质量(>=5): {high} 艘 ({high/len(boat_stats_df)*100:.1f}%)")
            print(f"  中等(2-4):   {medium} 艘 ({medium/len(boat_stats_df)*100:.1f}%)")
            print(f"  低质量(<2):  {low} 艘 ({low/len(boat_stats_df)*100:.1f}%)")
            
            return boat_stats_df
        else:
            print("[警告] 没有找到有BoatID的ESM数据")
            return None
    
    def analyze_2d_radar_matching_task(self, test_file='official/test/radar_detection.csv'):
        """分析2D雷达匹配任务"""
        print("\n" + "="*70)
        print("[2D雷达匹配任务分析]")
        print("="*70)
        
        df = pd.read_csv(test_file)
        radar_2d = df[df['Distance'] > 0].copy()
        
        radar_no_id = radar_2d[radar_2d['BoatID'] == -1]
        radar_with_id = radar_2d[radar_2d['BoatID'] != -1]
        
        print(f"  需要匹配的2D雷达点: {len(radar_no_id)}")
        print(f"  已有ID的2D雷达点: {len(radar_with_id)}")
        print(f"  匹配率: {len(radar_no_id)/len(radar_2d)*100:.1f}% 需要匹配")


def main():
    print("="*70)
    print("数据分层分析工具")
    print("="*70)
    
    analyzer = SimpleDataAnalyzer()
    
    # 分析训练集
    train_stats = analyzer.analyze_train_data()
    
    # 分析测试集
    test_stats = analyzer.analyze_test_data()
    
    # 分析匹配任务
    analyzer.analyze_2d_radar_matching_task()
    
    # 保存结果
    if train_stats is not None:
        output_dir = Path('analysis_results')
        output_dir.mkdir(exist_ok=True)
        
        train_file = output_dir / 'train_crossing_stats.csv'
        train_stats.to_csv(train_file, index=False)
        print(f"\n[训练集统计已保存]: {train_file}")
    
    if test_stats is not None:
        output_dir = Path('analysis_results')
        test_file = output_dir / 'test_crossing_stats.csv'
        test_stats.to_csv(test_file, index=False)
        print(f"[测试集统计已保存]: {test_file}")
    
    print("\n" + "="*70)
    print("[分析完成]")
    print("="*70)
    
    # 输出建议
    if train_stats is not None:
        print("\n[建议的分层阈值]")
        q75 = train_stats['CrossingPoints'].quantile(0.75)
        q25 = train_stats['CrossingPoints'].quantile(0.25)
        print(f"  高质量层: 交叉点 >= {int(q75)}")
        print(f"  中等质量层: 交叉点 {int(q25)}-{int(q75)-1}")
        print(f"  低质量层: 交叉点 < {int(q25)}")


if __name__ == '__main__':
    main()

