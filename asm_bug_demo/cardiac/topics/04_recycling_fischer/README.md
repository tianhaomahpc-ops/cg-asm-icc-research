# 回收 / Fischer(EP 时间循环)

跨时间步回收把重复求解成本摊掉的全部结果。

## 图

- `fig_fischer_eploop.png` — 真实 EP 循环:每步迭代数 cold vs Fischer
- `fig_fischer_demo.png` — Fischer 机制:A-正交投影(可复现)
- `fig_fischer_batched.png` — 大规模改造:打包+CGS2 前后对比

## 文档 / 幻灯片

- `fischer_eploop.txt`
- `fischer_before_iters.txt`
- `fischer_after_iters.txt`
