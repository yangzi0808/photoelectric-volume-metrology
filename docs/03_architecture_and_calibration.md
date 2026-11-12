# 03 代码架构、标定与误差预算

> 第二轮已在本文规划之上实现原图测量、双相机去畸变、顶部高度视差补偿、
> ChArUco 标定、激光平面标定和结果锁定。实际压测情况以
> [05_competitive_upgrade_and_validation.md](05_competitive_upgrade_and_validation.md) 为准。

## 1. 当前实现层级

```text
输入层
  top_mask         二值化顶部轮廓
  side_mask        二值化侧面轮廓
  depth_profile    激光深度样本
  calibration      顶部 x/y 尺度与侧面高度尺度
       |
几何特征层
  tight bbox
  corner notch
  enclosed hole
  two-level laser depth
       |
参数层
  L: a,b,h,u,v
  C: D,d,H
  B: a,b,h,u,v,t
       |
判定层
  shape mode / range / geometry checks
  first-order uncertainty
       |
输出层
  cm3 / quality issues / elapsed time / diagnostic JSON / PNG
```

代码入口是 `python -m volumecv`。各模块职责：

- `features.py`：从二值轮廓提取紧包围盒、角缺口、封闭内孔，并做辅助判型；
- `pipeline.py`：把像素量转换成毫米，调用已知公式和不确定度模型；
- `laser.py`：两级稳健聚类与离群点剔除；
- `geometry.py`：体积公式、规则范围和几何合法性；
- `benchmark.py`：随机尺寸、像素尺度、边缘膨胀/腐蚀、标定偏差和重复性测试；
- `reporting.py`：可追溯 JSON、CSV 和诊断图。

## 2. 规划中的硬件采集层

```text
background frame ─┐
top silhouette ───┼─> calibration / edge fitting ─> top parameters
side silhouette ──┤                              └> height
laser frames ─────┘─> line extraction -> ray/plane -> depth
```

建议新增以下模块：

```text
volumecv/capture.py
  capture_synchronized()
  lock_exposure()
  trigger_camera()
  set_backlight()
  set_laser()

volumecv/calibration.py
  calibrate_intrinsics()
  fit_top_plane_homography()
  fit_side_scale()
  fit_laser_plane()
  validate_calibration()

volumecv/edges.py
  refine_edge_subpixel()
  fit_lines_robust()
  fit_parallel_pair_distance()
  fit_circles_ransac()

volumecv/hardware/
  camera.py
  serial_controller.py
  state_machine.py
```

这些模块需要真实硬件后完成，不应以合成测试替代。

## 3. 标定流程

### 3.1 相机内参

1. 固定焦距、镜头、分辨率和 ROI；
2. 拍摄 20 至 30 个不同姿态的 ChArUco/棋盘格视图；
3. 用 `calibrateCamera` 求 `K、D`；
4. 检查总重投影 RMS、每视图误差和不同区域误差；
5. 保存标定版本、图像尺寸、曝光和镜头锁紧状态。

目标：重投影 RMS `<0.3 px`，不只看平均误差，还要检查边角视图。

### 3.2 顶部毫米平面

顶部被测物体只在固定工作平面附近，使用固定高度标定板：

1. 在标准板检测四个以上已知点；
2. 对去畸变图像求单应矩阵 `H_top`；
3. 把后续 ROI 变换到等尺度毫米网格；
4. 在网格上用已知长度块复测标定；
5. 保存逆映射用于原始像素留档。

正式测目标时不能拿 L 块或圆柱本身做标定；使用独立打印或机加工的标准图案/块。

### 3.3 侧面高度尺度

侧面只需恢复沿竖直方向的像素/毫米比例和底部基准：

1. 把已知高度的基准块放在同一测量台面；
2. 上下基准边做亚像素直线拟合，求像素间距；
3. 在多个高度验证线性；
4. 正式测量时用刚性基准边确定底部零位；
5. 检查相机、夹具和台面是否发生移动。

不能直接用靶标高度预存答案；标定块只提供尺度和基准。

### 3.4 激光平面

标准激光三角模型：

```text
pi: n^T X + d = 0
r(lambda) = C + lambda * v
lambda = -d / (n^T v)
X = C + lambda * v
```

其中 `C` 是相机光心，`v` 是去畸变像素对应的世界坐标射线方向，`n,d` 是激光平面。

标定方法：

1. 使用至少两个已知高度且相互平行的平面；
2. 激光在不同高度平面上的成像点反投影成三维点；
3. SVD 拟合激光平面；
4. 用一个未参与拟合的高度复查；
5. 记录残差、相机温漂和不同 ROI 的平面误差。

### 3.5 每日快速校验

比赛准备时间只有 10 分钟。建议准备一个固定标准块：

```text
顶部长度 50.000 mm
侧向高度 30.000 mm
一个 20.000 mm 的校准台阶
```

程序自动确认：

- 顶部尺度偏差 <0.15%；
- 侧面高度偏差 <0.15%；
- 激光台阶深度偏差 <0.2 mm；
- 相机姿态变化 <标定门限。

校验失败时给出明确原因，不允许“带着未知比例误差继续测”。

## 4. 当前 MVP 的相对误差估算

当前模型对每个长度采用：

```text
sigma_L = sqrt((sqrt(2)*sigma_edge/scale_px_per_mm)^2
               + (relative_scale_sigma*L)^2)
```

默认：

- 边缘定位 `sigma_edge = 0.35 px`；
- 尺度相对误差 `0.15%`；
- 激光系统深度误差 `0.03 mm`。

然后对体积公式做一阶独立误差传播。例如盲槽：

```text
dV/da = b*h
dV/db = a*h
dV/dh = a*b
dV/du = -v*t
dV/dv = -u*t
dV/dt = -u*v
```

这不是完整实测误差模型。真实系统还需加入：

- 镜头畸变残余；
- 标定板平面度；
- 角度装调误差；
- 背光边缘衍射/模糊；
- PLA 表面斜角、打印纹理和形状公差；
- 激光散斑与表面反射；
- 相机温漂、夹具蠕变；
- 目标放置位置变化；
- 亚像素拟合模型偏差。

## 5. 边缘算法升级要求

当前离线 MVP 从已二值化轮廓直接读取像素跨度。真实图像必须改成：

1. 对灰度图做亚像素边缘定位；
2. 丢弃短边、反光饱和和低梯度点；
3. 对 L 块每条外边界独立拟合直线；
4. 对缺口边界和内孔边界独立拟合；
5. 对圆柱用 RANSAC/鲁棒最小二乘拟合圆；
6. 转换到毫米平面后重新拟合，避免像素空间各向异性；
7. 输出拟合残差和有效边点比例，作为质量门限。

## 6. 结果锁定与可追溯性

每次测量的最小记录：

```text
timestamp
shape_mode
software_version
calibration_version
raw_top_image / raw_side_image / laser_frames
image_sha256
exposure / gain / illumination
fitted dimensions
volume_cm3
uncertainty
quality issues
elapsed_ms
first_display_timestamp
```

结果首次进入“有效且稳定”状态后锁定。此后只允许保存日志，不允许静默改体积。

## 7. 当前 MVP 的已知技术债

- 没有真实相机、SDK 或设备驱动；
- 输入是二值 mask，不是原图；
- L 块缺口假设为轴对齐角缺口；
- 圆柱使用包围盒初值，尚未使用圆形 RANSAC；
- 侧面高度使用紧包围盒，不含底边基准拟合；
- 盲槽深度输入是已提取的深度样本；
- 自动判型只针对三类合成形状；
- 误差传播未包含真实系统误差和相关性；
- 没有 GUI、异常恢复和硬件看门狗。

这些债必须在阶段 C 至 F 清除，才能把“算法演示”升级为“现场测量系统”。
