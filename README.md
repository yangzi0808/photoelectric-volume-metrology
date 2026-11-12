# 光电赛体积测量系统

面向 2026 深圳大学物光创新赛赛题二的比赛原型。主方案为：

> **顶部背光轮廓 + 固定侧视测高 + 毫米平面标定 + 局部线激光测深**

已知形状下直接测量几何参数，再代入官方体积公式，避免把 8 分钟耗在不必要的
稠密点云重建上。

## 当前状态

已实现：

- 原图分割、轮廓提取和自动判型；
- 顶部/侧面透视单应到毫米平面；
- L 块六边轮廓、凹角、缺口亚像素边缘拟合；
- 通孔圆柱内外圆稳健拟合与径向亚像素质心；
- 矩形槽口四边直线拟合；
- 侧面上下边界亚像素拟合；
- 根据实测高度补偿顶面视差，解决台面标定与物顶高度不同导致的尺度误差；
- 线激光中心提取、相机射线-激光平面求交、已知高度平面标定；
- 两级槽深稳健估计；
- ChArUco 内参、顶部平面、侧面平面和激光平面标定；
- 双相机采集接口、实时测量 CLI、8 分钟状态机和结果锁定门限；
- JSON/CSV/图卡输出及合成压力基准。

尚未完成：

- 真实相机、灯箱、夹具和激光器的硬件实测；
- 官方目标或同尺寸替代目标的最终验收；
- 温漂、跌落、现场杂散光和连续运行测试；
- 国一级别不能由仿真承诺，必须由实物重复测量数据兑现。

## 快速运行

使用项目内的 Python 3.12 环境：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m volumecv image-demo --out output\image_demo
```

如果 `.venv` 不存在：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

主要输出：

- `output/image_demo/dashboard.png`：原图链路诊断图；
- `output/image_demo/report.json`：每次尺寸、体积、误差、置信度；
- `output/image_demo/input_images/`：可复核的顶部/侧面测试图像。

运行 120 组以上的真实图像压力基准：

```powershell
.\.venv\Scripts\python.exe -m volumecv image-benchmark `
  --per-shape 20 --repeats 2 --seed 20261110 `
  --out output\image_benchmark_final
```

最终合并汇总位于 `output/benchmark_combined/summary.json`，高度视差专项位于
`output/parallax_benchmark/parallax_summary.json`。

## 真实硬件校准

首先生成标定板并打印在刚性平板上：

```powershell
.\.venv\Scripts\python.exe -m volumecv generate-charuco --out configs\charuco.png
```

打印时必须选择 100% 实际尺寸，不得使用“适应页面”；打印后用钢尺复核方格边长
20.000 mm，并把标定板粘在平整刚性板上。

约定：

- 7 x 5 方格；
- 方格边长 20.000 mm；
- ArUco 边长 15.000 mm；
- `DICT_6X6_250`。

拍摄 20 至 30 张不同姿态的顶部相机标定图，放到
`data/calibration/top/`。侧面相机使用另一台相机时，放进
`data/calibration/side/`。再把标定板分别放在测量台面和侧视平面拍摄：

```powershell
.\.venv\Scripts\python.exe -m volumecv calibrate `
  --images data\calibration\top `
  --side-images data\calibration\side `
  --top-image data\calibration\top_plane.png `
  --side-image data\calibration\side_plane.png `
  --out configs\calibration.json
```

`--side-images` 可省略，此时侧面复用顶部相机内参，仅适合同一台相机。

标定验收门限：

- 相机重投影 RMS `<0.3 px`，绝对不可超过 `0.5 px`；
- 顶部/侧面平面单应 RMS `<0.2 mm`；
- 激光平面 RMS `<0.2 mm`；
- 用独立标准块复测长度误差 `<0.2%`。

## 真实图片测量

```powershell
.\.venv\Scripts\python.exe -m volumecv measure-images `
  --calibration configs\calibration.json `
  --top-image data\top.png `
  --side-image data\side.png `
  --shape cylinder `
  --out output\real_cylinder.json
```

盲槽增加激光深度样本文件：

```powershell
.\.venv\Scripts\python.exe -m volumecv measure-images `
  --calibration configs\calibration.json `
  --top-image data\top.png `
  --side-image data\side.png `
  --depth-file data\slot_depth.txt `
  --shape blind_slot
```

## 双相机现场采集

先探测 Windows 相机编号：

```powershell
.\.venv\Scripts\python.exe -m volumecv camera-list
```

完成标定后：

```powershell
.\.venv\Scripts\python.exe -m volumecv live-measure `
  --calibration configs\calibration.json `
  --top-camera 0 `
  --side-camera 1 `
  --shape auto `
  --frames 8 `
  --out output\live
```

也可以打开双相机桌面界面：

```powershell
.\.venv\Scripts\python.exe -m volumecv gui
```

第一次安装时必须人工检查：

- 曝光是否锁定；
- 自动白平衡和自动对焦是否关闭；
- 帧率是否稳定；
- 顶部相机是否垂直；
- 标定后轻碰夹具，已知长度是否仍复现；
- 激光开/关是否与采集同步。

## 体积模型

```text
L 形块          V = (a*b - u*v) * h
通孔圆柱        V = pi/4 * (D^2 - d^2) * H
矩形盲槽块      V = a*b*h - u*v*t
```

内部尺寸均为 mm，最终体积为 cm³。

## 已完成的合成验证

默认 120 次原图压力测试包含：

- 0–100 mm 范围内的随机尺寸；
- 透视、缩放、平移和像素级旋转；
- 高斯模糊、亮度梯度、暗角和传感器噪声；
- JPEG 压缩；
- 顶部/侧面独立标定小偏差；
- 激光噪声和离群点；
- 同一目标两次独立采集。

固定种子测试结果：

```text
平均绝对误差     0.154%
p95 绝对误差     0.343%
最大绝对误差     0.505%
5% 门槛通过率    100%
判型准确率       100%
两次差异 p95     0.333%
2σ 区间覆盖率    97.5%
```

另用两个不同随机种子分别做 72 次和 60 次交叉验证，最大误差分别为 `0.376%`
和 `0.428%`，判型准确率均为 `100%`。三个种子合并共 252 次，平均误差
`0.148%`、p95 `0.345%`、最大 `0.505%`，2σ 覆盖 `97.2%`。

顶部高度视差专项 15 次：不补偿平均误差 `74.46%`；补偿后平均 `0.251%`、
p95 `0.431%`、最大 `0.458%`。

这些结果显示算法链路有继续实物化的价值，但不能替代真实标定误差、打印公差、
反光、温漂、机械刚度和现场条件测试。

## 项目结构

```text
volumecv/
  calibration.py      ChArUco、内参、单应、激光平面、高度补偿
  camera.py           双相机采集、曝光/增益设置、多帧平均
  gui.py              双相机桌面测量与结果锁定界面
  image_measure.py    原图分割、亚像素边缘、直线/圆拟合
  image_pipeline.py   高度补偿后的尺寸与体积闭环
  realistic.py        透视/模糊/噪声/JPEG 仿真
  laser_scan.py       激光中心线提取和三角化
  laser_measure.py    平面位姿、激光标定、槽深估计
  session.py          8 分钟流程和结果锁定
  geometry.py         体积公式、规则校验
  uncertainty.py      一阶误差传播
  cli.py              命令行入口
tests/                 28 个单元/集成测试
docs/                  调研、方案、标定与复盘
```

## 比赛边界

- 不使用商用扫描仪、工业轮廓仪、深度相机输出的深度图/点云/体积；
- 不调用厂商 SDK 直接输出尺寸或体积；
- 标定只使用自备的已知标准件，不拿正式目标预存答案；
- 不接触目标，不人工描边、读取关键尺寸或修正结果；
- 激光必须采用低功率、遮光、安全标识齐全的方案；
- 结果首次锁定后保留原始图像、标定版本、尺寸、软件版本和用时。

详细方案见 [docs/01_related_work.md](docs/01_related_work.md)、
[docs/02_solution_and_plan.md](docs/02_solution_and_plan.md) 和
[docs/03_architecture_and_calibration.md](docs/03_architecture_and_calibration.md)。
第二轮代码审计与压测记录见
[docs/05_competitive_upgrade_and_validation.md](docs/05_competitive_upgrade_and_validation.md)。
