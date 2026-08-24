# learnable_sde OOP 架构设计

状态：目标架构基线（v1）
适用范围：模型、估计、推理、数据、评价与实验编排
实现策略：直接重构当前顶层 `models/`、`estimation/`、`inference/`、`data/` 与
`experiments/`；新增 `domain/`、`application/`、`evaluation/`、`infrastructure/`
和 `cli/` 承担缺失职责。迁移期间不得改变既有数学行为。

## 1. 设计目标

本项目需要提供一个可组合、可测试、可复现的 SDE 研究框架。框架应支持：

1. 加载轨迹、条件特征和 checkpoint；
2. 表达线性、隐模式和神经 SDE；
3. 通过 EM、NLL、proper scoring rule 或迁移学习估计模型；
4. 通过精确核、数值积分、Monte Carlo、Fokker--Planck 或 bridge 完成推理；
5. 在相同数据切分与随机流上完成评价、消融和报告；
6. 新增一种模型、估计器或推理器时，不修改既有核心流程。

### 1.1 非功能需求

- 相同配置、数据和 seed 产生相同结果；
- CPU/GPU 与 dtype/device 生命周期由模型和运行上下文显式管理；
- 数据错误、配置错误、能力不匹配和数值失败必须尽早报告；
- 不允许未声明的 baseline fallback；
- 模型使用标准 `torch.nn.Module` 生命周期及 `state_dict`；
- 核心模块不得依赖模块级可变缓存、最近一次调用状态或全局 RNG；
- 训练、推理和评价之间只传递有名字、有形状约束的领域对象。

### 1.2 非目标

v1 不负责分布式训练、GUI、远程任务调度或算法本身的重新推导。迁移阶段优先保持数值等价，再逐项修复已知正确性问题。

## 2. 核心用例

### 2.1 训练

输入为类型化配置与训练轨迹；输出为已更新模型、`FitResult` 和 checkpoint。估计器可以更新传入模型，但不能直接依赖模型私有字段。

### 2.2 预测

输入为已训练模型与 `ForecastRequest`；显式选择推理器；输出统一 `Forecast`。不支持的模型/推理组合抛出 `CapabilityError`。

### 2.3 评价

输入为 Forecast、Observation 与 ScoringRule；输出逐段分数、汇总分数和不确定性估计。评分口径只能有一处规范实现。

### 2.4 消融

每条实验臂共享数据切分和随机流，独立装配 model、estimator、inference 和 evaluation。失败、跳过与不支持必须进入结果，不得伪装成基线结果。

## 3. 领域模型

跨模块只传递以下稳定对象：

| 对象 | 责任 |
|---|---|
| `TrajectorySegment` | 一段带时间、状态、可选条件和元数据的轨迹 |
| `TrajectoryDataset` | train/validation/evaluation 三个无泄漏切分 |
| `TransitionBatch` | 一步转移 `(x, y, dt, condition)` |
| `ModelContext` | 单次模型调用所需的 condition 与 regime |
| `GaussianTransition` | 精确高斯核的 mean/covariance |
| `ForecastRequest` | 初态、horizons、样本数和模型上下文 |
| `Forecast` | 样本及可选解析矩 |
| `FitResult` | 收敛状态、目标历史和诊断 |

数据对象默认不可变，并在进入算法边界前执行形状、有限性和时间顺序校验。

## 4. 模型接口与能力隔离

### 4.1 基础模型

所有动力学模型实现 `SDEModel(torch.nn.Module)`：

```python
class SDEModel(torch.nn.Module, ABC):
    @property
    @abstractmethod
    def state_dim(self) -> int: ...

    @property
    @abstractmethod
    def noise_dim(self) -> int: ...

    @abstractmethod
    def drift(self, t, x, context: ModelContext): ...

    @abstractmethod
    def diffusion(self, t, x, context: ModelContext): ...
```

所有实现保持相同签名。隐模式通过 `ModelContext.regime` 显式传递，禁止 `_active_mode` 一类对象内隐藏状态。

### 4.2 能力接口

不是每个 SDE 都具有相同能力，因此不在大基类中提供返回 `None` 的可选方法：

- `ExactTransitionProvider`：提供 `exact_transition()`；
- `LatentRegimeModel`：提供隐模式似然、先验和 EM 更新入口；
- `ParameterGroupProvider`：为微调提供具名参数组。

例如，线性隐模式模型可以同时实现三个能力；神经 SDE 只需实现基础模型和参数分组能力。精确推理器依赖 `ExactTransitionProvider`，而不是依赖某个具体模型类。

### 4.3 参数所有权

模型拥有参数及参数约束。估计器产生更新或充分统计量，由模型公开方法应用。禁止外部通过 `.data = ...`、私有矩阵方法或字段名修改模型。

非梯度更新使用 `torch.no_grad()` 和 `Tensor.copy_()`；保存和恢复只使用 `state_dict()`。

## 5. 数据与持久化接口

数据源按返回类型泛型化：

```python
class DataSource(Generic[T], ABC):
    @abstractmethod
    def load(self) -> T: ...
```

- `TrajectorySource` 返回 `TrajectoryDataset`；
- `ConditionProvider.features_for(segment)` 按段提供条件特征；
- `ModelStore` 专门负责模型 checkpoint；
- `ArtifactStore` 专门负责配置、指标和报告。

Checkpoint 和条件特征不伪装成轨迹数据源。Parquet、文件路径与缓存属于 infrastructure，不进入 domain 或模型。

## 6. 估计接口

```python
class Estimator(Generic[ModelT, DataT], ABC):
    @abstractmethod
    def fit(
        self,
        model: ModelT,
        data: DataT,
        context: FitContext,
    ) -> FitResult: ...
```

模型在 `fit()` 显式传入，估计器配置保存在估计器对象中。具体实现可以限定所需模型能力和数据类型。例如 `SegmentEMEstimator` 依赖 `LatentRegimeModel`，而 `NLLEstimator` 消费 `TransitionBatch`。

`FitResult` 统一表达收敛、迭代数、目标历史和诊断；算法正常结束但未收敛不等同于运行异常。

## 7. 推理与评价接口

```python
class InferenceEngine(ABC):
    @abstractmethod
    def supports(self, model: SDEModel) -> bool: ...

    @abstractmethod
    def forecast(
        self,
        model: SDEModel,
        request: ForecastRequest,
        context: InferenceContext,
    ) -> Forecast: ...
```

计划实现 `ExactGaussianEngine`、`EulerMaruyamaEngine`、`SplitStepEngine`、`MonteCarloEngine`、`FokkerPlanckEngine` 和 `BridgeEngine`。路由由配置明确选择；`supports()` 失败时立即报错。

评分器统一为：

```python
class ScoringRule(ABC):
    @abstractmethod
    def score(self, forecast: Forecast, observation: Tensor) -> Tensor: ...
```

Energy Score 在新架构中只采用

`E||X-y|| - 0.5 E||X-X'||`

这一规范定义。历史口径如需保留，必须命名为独立 legacy 实现，不能通过含义模糊的布尔参数切换。

## 8. 配置、装配与对象生命周期

`ExperimentConfig` 由 runtime、data、model、estimator、inference、evaluation 和 output 七个类型化配置组成。每个组件类型使用可区分配置，不传递自由结构的嵌套字典。

Registry 是 application 层拥有的实例对象：

```python
models.register("segment_regime", build_segment_regime)
estimators.register("em", build_em)
engines.register("exact", build_exact_engine)
```

Registry 不设置全局 seed，不运行算法，也不静默返回默认组件。应用入口是唯一 composition root。

`RandomStreams` 从主 seed 派生 training、inference 和 bootstrap 三个局部 `torch.Generator`。缓存由 repository 或显式 `RunContext` 拥有，随一次应用运行创建和释放。

## 9. 核心流程

```text
配置解析与校验
    -> 组件装配与能力检查
    -> 数据加载、切分与校验
    -> 特征转换
    -> Estimator.fit
    -> ModelStore.save
    -> InferenceEngine.forecast
    -> Evaluator.evaluate
    -> ArtifactStore.write
```

训练、预测和评价由 `ExperimentApplication` 编排。CLI 只解析参数并调用应用服务，不能包含算法或数据分支。

## 10. 异常规范

统一异常层级：

- `ConfigurationError`：配置格式、未知组件或重复注册；
- `DataValidationError`：数据形状、有限性或时间顺序错误；
- `CapabilityError`：组件组合不兼容；
- `NumericalError`：矩阵分解等数值过程无法恢复；
- `ConvergenceError`：算法无法继续迭代。

禁止 `except Exception: pass`。可恢复降级必须记录具体异常、策略和结果 metadata。

## 11. 目标目录与依赖方向

```text
项目根目录/
├── domain/          # 领域对象和异常，无 pandas/CLI/路径依赖
├── models/          # SDEModel 与能力接口、具体模型
├── estimation/      # estimator 接口和算法
├── inference/       # inference 接口和算法
├── evaluation/      # scoring 与 uncertainty
├── data/            # source、condition、validation、transform
├── infrastructure/  # parquet、checkpoint、cache、artifact
└── application/     # config、registry、compatibility、experiment
```

依赖只能向内：CLI/infrastructure/application 可以依赖领域与算法接口；domain 不得依赖外层模块；`experiments/` 最终只保留薄入口。

## 12. 测试标准

### 12.1 契约测试

每个模型实现都必须通过同一组测试：

- drift/diffusion 形状；
- batch、dtype 和 device 保持；
- 梯度可传播；
- `state_dict` round trip；
- context 缺失时明确失败；
- 不污染全局 RNG。

精确核实现还需验证 covariance 对称/正半定、`dt=0` 恒等和与高精度数值解对拍。

### 12.2 集成测试

- 配置到训练再到 checkpoint；
- checkpoint 到预测再到评价；
- 不兼容组合明确失败；
- 两次完整运行可复现；
- 数据切分无泄漏。

### 12.3 数值回归

保留现有精确核、EM、积分器和评分 gate，采用具有数学意义的容差。数值 gate 与 OOP 契约测试分层，避免用数值快照代替接口验证。

## 13. 迁移计划

1. 建立领域对象、异常和契约测试；
2. 让目标 `SDEModel` 基于 `nn.Module`，统一 `ModelContext`；
3. 迁移 `SegmentConstantSDE`，保持现有数值 gate；
4. 迁移 EM，消除对模型私有字段及 `.data` 的访问；
5. 建立类型化配置和 model/estimator/inference 三类 registry；
6. 建立统一 inference 与 evaluation；
7. 将旧 `backend.py` 拆为 repository、fit service、forecast service 和 mechanism evaluator；
8. 最后接入神经 SDE、迁移学习和高级推理；
9. 所有调用切换完成后删除 legacy 模块。

每一步必须同时满足旧数值 gate 与新契约测试。迁移期间不在同一个变更中重写算法、修正科学口径和移动架构。

## 14. 完成定义

当以下条件全部成立时，OOP 主干才算完成：

1. 新增模型只需实现模型/能力接口并注册；
2. 新增估计器或推理器不修改模型、数据层和实验主流程；
3. 所有模型支持标准 PyTorch 生命周期；
4. 核心流程没有模块级共享可变状态；
5. 不支持的组合不会静默落入 baseline；
6. 训练、预测、评价和 checkpoint 具有端到端契约测试；
7. 现有数值回归继续通过。

## 15. CLI 契约

CLI 是 application service 的薄适配层，不包含模型分支或拟合算法：

```bash
# 训练：合成 smoke 或真实 split，可选保存标准 state_dict checkpoint
python -m cli train --config config.yaml --smoke --checkpoint outputs/model.pt
python -m cli train --config config.yaml --split train --max-segments 100

# 预测：显式 checkpoint、初态、horizons、样本数和 latent regime
python -m cli predict --config config.yaml --checkpoint outputs/model.pt \
  --x0 0 0 --horizons 60 120 --samples 100 --regime 0

# 消融/数值门
python -m cli ablate --verify
python -m cli ablate --matrix
```

`experiments/run.py` 只保留到 `cli.train` 的兼容入口。所有 CLI 返回进程状态码并输出
结构化 JSON；预测样本仅在显式 `--output` 时写盘。安装项目后，`pyproject.toml`
同时提供等价的 `learnable-sde train|predict|ablate` console script。

## 16. 当前落地状态

已进入真实执行路径：

- domain 对象与异常；canonical `TrajectorySegment` 已由 loader 实际产出；
- `SDEModel(nn.Module)`、精确核/隐模式/参数分组能力和 `ModelContext`；
- `SegmentConstantSDE`、类型化 `SegmentEMData`、显式 `FitContext` 与 `FitResult`；
- exact、Euler、J-1 split、J-3 CRN inference engine；
- Energy Score/CRPS evaluation service；
- model/estimator/inference registry、ExperimentApplication、checkpoint/artifact adapter；
- train/predict/ablate CLI 与 console script；
- 契约、application、checkpoint、backend 和数值回归测试。

仍属于算法缺口而非 OOP 接线：

- `TimeVaryingNeuralSDE` 尚无实际 drift/diffusion 网络；因此没有注册 neural model；
- J-2 目前只有 1D 收敛内核，2D FP solver 仍是占位，因此没有注册 `J2_FP` engine；
- `experiments/backend.py` 保留外部 22 臂兼容契约和 `_Context`，核心调用已迁移到新模型/EM
  接口，但物理拆包要等正式 22 臂基线可回放后单独进行。

这些组合选择时必须 fail-fast，不得静默映射到已经实现的组件。
