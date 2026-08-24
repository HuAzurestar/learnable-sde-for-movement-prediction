"""tests —— pytest 回归/特征化测试。

运行：python -m pytest（自动发现 tests/ 下的 test_*.py）。

BLAS 顺序护栏在 conftest.py 置顶（numpy/pandas 先于 torch 装载）。
"""
