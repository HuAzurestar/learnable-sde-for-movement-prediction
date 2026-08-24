"""数值常量集中 —— 命名并文档化 float64 关键路径的数值小量。

这些小量原为散落各文件的魔法数字，此处是它们的单一事实源。逐处替换**同值**字面量
以提升可读性（值不变 → 行为不变）。当前已完成协方差 Cholesky 抖动的迁移
（models/base.py、models/segment_constant.py），其余常量作为后续迁移的规范值。
"""

from __future__ import annotations

# 协方差 Cholesky 分解前的对角抖动（数值稳定，float64 尺度）。
CHOLESKY_JITTER = 1e-12

# 残差协方差正则（np.cov 前加的 ε，避免奇异；见 backend.py Batch3 GMM）。
COV_JITTER = 1e-9

# log 域 epsilon（softmax / prior_logits / 概率）。
LOG_EPS = 1e-12

# 行列式 / 除零守卫。
DET_EPS = 1e-12

# 最小方差 / 最小扩散平方（方差下界）。
MIN_VAR = 1e-12
MIN_G2 = 1e-12


def safe_cholesky(S, jitter: float = CHOLESKY_JITTER):
    """Cholesky 分解 + 对角抖动，返回下三角 L。

    为什么：协方差 S 常为半正定（数值/退化），直接 `cholesky(S)` 会因最小特征值
    为 0 或负而报错；加 `jitter * I` 抬高最小特征值，保证可分解且对 float64 尺度
    误差可忽略。全库 13 处 `cholesky(S + 1e-12*eye)` 收敛到这一处。
    """
    import torch

    d = S.shape[-1]
    eye = torch.eye(d, dtype=S.dtype, device=S.device)
    return torch.linalg.cholesky(S + jitter * eye)
