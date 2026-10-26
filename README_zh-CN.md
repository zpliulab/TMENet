# TMENet

TMENet联合估计细胞类型特异的基因调控网络（GRN）和空间细胞耦合；TMENet-LR再将耦合权重归因到表达支持的配体–受体候选对。结合数据库受体–TF路径、接收细胞共同检出及推断GRN，可以进一步优先排序候选跨细胞调控路径。

## 安装与运行

在本文件所在的仓库根目录执行，推荐Python 3.10及以上：

```bash
python -m pip install .
tmenet fit --input examples/toy/inputs.npz --config configs/toy.json --output results/toy --backend cpu
tmenet lr --expression examples/toy/expression.tsv --coupling results/toy/C.npy --metadata examples/toy/metadata.tsv --pairs examples/toy/lr_pairs.tsv --mask examples/toy/candidate_mask.npy --output results/toy_lr.tsv
```

第一次运行建议先使用CPU。GPU需要另行安装与CUDA环境匹配的CuPy，运行时改为`--backend gpu`。不会在GPU不可用时静默冒充GPU运行。

`results/toy/`会同时保存B、C、X、标签、实际参数、目标函数和停止原因。达到最大迭代次数的有限结果照常保存，但不标记为已经收敛。为防止误覆盖，重新运行时请使用新的输出目录。

## 输入与方法

- 输入格式见[INPUT_OUTPUT.md](docs/INPUT_OUTPUT.md)。表达矩阵为基因×空间点；metadata必须有`cell`、`cell_type`两列；坐标与空间点按标签对齐。
- ATAC先验可由peak共同可及性、基因/TSS注释及motif整合后得到TF–target候选边，再转成二值先验。没有ATAC时可以使用自行准备的数据库先验。
- 新接口中的B为“target行、regulator列”；C为“sender行、receiver列”。不要把C直接解释成LR相互作用概率。
- 默认按Gabriel图平均度自适应选择K，并保留第K个邻居的距离并列点，不固定为K=4或K=20。
- 支持六种p；非整数p使用正式六p实验的近端算子。详见[METHODS.md](docs/METHODS.md)。

## 论文复现

`configs/ami.json`、`configs/hcc.json`和`configs/simulation_strata.json`保留相应参数。HCC使用修复后的加权投影及2,000次预算；不能直接套用AMI配置。完整复现需要原始公开数据、正确的样本/细胞类型匹配及下游数据库。

`reproducibility/`提供模拟生成、统一GRN/LR评分、baseline接口、HCC PCA-CMI修复及消融工具。第三方baseline需要各自的R/Python环境，不属于核心安装依赖。

示例数据仅用于验证接口，不是论文的性能结果。


