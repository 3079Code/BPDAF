# BPDAF

论文：**Baseline-Preserving Distance-Aware Adaptive Fusion for LiDAR-Camera 3D Object Detection**

作者：Jiaxin Yang、Yongbiao Li、Zhanlin Cao、Long Chen、Jinglong Wang

[English](README.md) · [方法说明](docs/method.md) · [接入说明](docs/integration.md) · [测试记录](docs/validation.md) · [复现状态](docs/reproduction.md)

BPDAF 在已经对齐的相机与激光雷达 BEV 特征上，保留原卷积融合路径，并增加由距离和特征能量共同引导的自适应残差分支。残差输出卷积采用零初始化，因此模块初始输出与原融合路径一致。训练过程中，两条路径均可更新。

## 当前版本的范围

本仓库是**依据论文重新编写的 PyTorch 实现**，包括融合模块、测试、合成数据训练示例和可选的检测框架接入适配器。它不是产生论文实验结果的原始实验代码，目前没有重新完成 nuScenes 或 KITTI 的整套检测训练与评测。

当前不包含完整检测器、数据集、预训练权重、完整实验配置及论文结果对应的训练日志。缺失项见[复现状态说明](docs/reproduction.md)。独立融合模块及单元测试只依赖 PyTorch，不要求安装 CUDA 自定义算子或完整检测框架。

## 安装和运行

要求 Python ≥ 3.10、PyTorch ≥ 2.1。先根据设备安装合适的 [PyTorch 版本](https://pytorch.org/get-started/locally/)，然后在仓库根目录运行：

```bash
python -m pip install -e ".[dev]"
python -m pytest
python examples/smoke_train.py
```

需要同时验证模型保存和读取时，运行
`python examples/smoke_train.py --output runs/smoke.pt`。

示例使用合成特征验证前向传播、反向传播和参数更新；其中的合成损失不是论文中的目标检测损失，示例结果不代表检测精度。

CI 配置覆盖 CPU 上的 Python 3.10 / PyTorch 2.1.2 和 Python 3.12 / PyTorch 2.6.0。完整检测框架的版本兼容性需要另外验证。

## 最小使用示例

```python
import torch

from bpdaf import BPDAFFuser

fuser = BPDAFFuser(
    in_channels=(80, 256),
    out_channels=256,
    bev_range=(-54.0, -54.0, 54.0, 54.0),
)

# 输入顺序：相机、激光雷达；两者已经对齐到同一 BEV 网格。
camera = torch.randn(2, 80, 24, 24)
lidar = torch.randn(2, 256, 24, 24)
output, auxiliary = fuser.forward_with_aux([camera, lidar])

assert output.shape == (2, 256, 24, 24)
torch.testing.assert_close(output, auxiliary["reference"], rtol=0, atol=0)
```

普通 `fuser([camera, lidar])` 只返回融合特征。`forward_with_aux` 额外返回 `reference`、`residual`、`weights` 和 `context`，可用于检查初始等价性和观察残差门控。

`bev_range` 的顺序是 `(xmin, ymin, xmax, ymax)`，单位为米；张量形状为 `(N, C, H, W)`，行对应 y 轴，列对应 x 轴。示例的 24 × 24 仅用于快速运行；论文的 nuScenes 融合网格为 180 × 180。KITTI 应填入真实融合特征对应的坐标范围，不能直接沿用 nuScenes 参数。

已有兼容参考融合模块时，可以用 `BPDAFFuser.from_reference(reference, bev_range=...)` 复制其参数和归一化状态。完整检测器权重需要按[接入说明](docs/integration.md)处理，不能假定任何外部检查点均可直接加载。

## 实现对应关系

| 论文设计 | 当前实现 |
| --- | --- |
| 原始融合路径 | 3 × 3 卷积、批归一化、ReLU，参数保持可训练 |
| 距离编码 | 网格中心到传感器原点的归一化半径及 4 组正余弦，共 9 通道 |
| 能量编码 | 两模态特征绝对值的通道均值，按空间均值归一化，截断到 [0, 5]，默认停止梯度 |
| 竞争门控 | 11 → 32 → 2 的点卷积，使用 softmax 得到互补权重 |
| 残差校正 | 对两模态线性投影、加权、拼接，再用 3 × 3 卷积得到残差 |
| 初始等价性 | 残差输出卷积置零，初始融合结果与参考路径一致 |

“Baseline-preserving”仅表示初始化时的输出等价，不表示参考参数冻结，也不保证训练后输出不变。门控权重只描述残差分支内的特征分配，不应解释为经过校准的传感器置信度。

更完整的公式及两层零初始化带来的梯度传播顺序见[方法说明](docs/method.md)。

## 开发检查

```bash
python -m ruff check .
python -m ruff format --check .
python -m pytest
python -m build
```

提交修改前请阅读 [CONTRIBUTING.md](CONTRIBUTING.md)。新的实验结论应同时提供配置、检查点、评测协议和日志。

## 引用及许可

引用信息见 [CITATION.cff](CITATION.cff)。论文的正式出版信息和 DOI 将在确认后补充。代码仓库：[3079Code/BPDAF](https://github.com/3079Code/BPDAF)。

本实现采用 [MIT 许可证](LICENSE)。外部检测框架和数据集继续遵循其各自的许可与使用条件。
