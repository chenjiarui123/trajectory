# 复赛轨迹重建项目

## 项目说明
基于稀疏侦测数据的船舶轨迹重建（复赛版本）

## 项目结构
```
trajectory/
├── ans/                    # 参考方案代码
├── official/              # 官方数据和工具
├── 0_plan/                # 实验代码
├── common/                # 公共模块（待创建）
└── methods/               # 各方案实现（待创建）
```

## 开发进度
- [x] 分析ans方案
- [x] 制定复赛策略
- [ ] 实现传感器误差估计
- [ ] Docker化
- [ ] 提交测试

## 团队成员
- 开发：[@chenjiarui123](https://github.com/chenjiarui123)
- 管理：学姐
- 协助：同学A

## 快速开始
```bash
# 克隆仓库
git clone https://github.com/chenjiarui123/trajectory.git

# 安装依赖（待补充）
pip install -r requirements.txt

# 运行（待补充）
python traj_rec.py
```

## 复赛关键点
1. **传感器误差**：±1°系统性偏差
2. **5号ESM**：唯一准确的传感器（用于校准）
3. **Docker提交**：运行时间<1小时

## 更新日志
- 2025-11-04: 初始化项目

