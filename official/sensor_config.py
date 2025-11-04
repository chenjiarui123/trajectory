"""
传感器配置文件
包含10个传感器的真实位置和参数
"""

# 传感器完整信息
SENSOR_INFO = {
    1:  {
        'type': 'ESM',
        'lon': 122.38713,
        'lat': 36.894102,
        'period_min': 10,      # 侦测周期（分钟）
        'range_km': 200,        # 基程范围（公里）
        'resolution_deg': 0.001 # 方位分辨率（度）
    },
    2:  {
        'type': '2D_Radar',
        'lon': 122.30758,
        'lat': 36.838530,
        'period_min': 8,
        'range_km': 160,
        'resolution_deg': 0.005
    },
    3:  {
        'type': '2D_Radar',
        'lon': 121.61980,
        'lat': 36.746284,
        'period_min': 10,
        'range_km': 160,
        'resolution_deg': 0.005
    },
    4:  {
        'type': '2D_Radar',
        'lon': 120.90122,
        'lat': 36.420567,
        'period_min': 6,
        'range_km': 160,
        'resolution_deg': 0.005
    },
    5:  {
        'type': 'ESM',
        'lon': 120.68410,
        'lat': 36.133295,
        'period_min': 10,
        'range_km': 200,
        'resolution_deg': 0.001
    },
    6:  {
        'type': '2D_Radar',
        'lon': 120.17829,
        'lat': 35.751984,
        'period_min': 6,
        'range_km': 160,
        'resolution_deg': 0.005
    },
    7:  {
        'type': '2D_Radar',
        'lon': 119.63001,
        'lat': 35.547414,
        'period_min': 10,
        'range_km': 160,
        'resolution_deg': 0.005
    },
    8:  {
        'type': '2D_Radar',
        'lon': 119.34929,
        'lat': 35.124468,
        'period_min': 6,
        'range_km': 160,
        'resolution_deg': 0.005
    },
    9:  {
        'type': '2D_Radar',
        'lon': 119.48342,
        'lat': 34.757768,
        'period_min': 8,
        'range_km': 160,
        'resolution_deg': 0.005
    },
    10: {
        'type': 'ESM',
        'lon': 119.47792,
        'lat': 34.708298,
        'period_min': 10,
        'range_km': 200,
        'resolution_deg': 0.001
    },
}

# 传感器位置字典（用于快速访问）
SENSOR_POSITIONS = {
    sid: (info['lon'], info['lat']) 
    for sid, info in SENSOR_INFO.items()
}

# ESM传感器ID列表
ESM_SENSORS = [sid for sid, info in SENSOR_INFO.items() if info['type'] == 'ESM']

# 2D雷达传感器ID列表
RADAR_2D_SENSORS = [sid for sid, info in SENSOR_INFO.items() if info['type'] == '2D_Radar']

# 打印传感器信息（当直接运行此文件时）
if __name__ == '__main__':
    print("=" * 90)
    print("传感器配置信息")
    print("=" * 90)
    print(f"{'编号':<6} {'类型':<12} {'经度':<12} {'纬度':<12} {'周期(min)':<12} {'范围(km)':<12} {'精度(°)':<12}")
    print("-" * 90)
    
    for sid, info in SENSOR_INFO.items():
        print(f"{sid:<6} {info['type']:<12} {info['lon']:<12.5f} {info['lat']:<12.6f} "
              f"{info['period_min']:<12} {info['range_km']:<12} {info['resolution_deg']:<12.3f}")
    
    print("=" * 90)
    print(f"\nESM传感器: {ESM_SENSORS}")
    print(f"2D雷达传感器: {RADAR_2D_SENSORS}")
    print(f"\n总计: {len(SENSOR_INFO)} 个传感器")
    print(f"  - ESM: {len(ESM_SENSORS)} 个 (仅角度信息)")
    print(f"  - 2D雷达: {len(RADAR_2D_SENSORS)} 个 (距离+角度信息)")

