# 假几何+真实心脏参数:ASM vs sASM 结果(汇报用)

cardiac-sim-fakegeo 分支的主结果:三系统耦合前向 ECG,真几何 FEM,ASM(BASIC) vs sASM 的迭代数与时间。**汇报 slides 在此。**

## 图

- `fig_mesh.png` — conforming 心脏嵌躯干网格
- `fig_meshview.png` — 网格切面
- `fig_geom.png` — 几何/子域分区
- `fig_vm.png` — Sys1 单域 Vm 场
- `fig_ue.png` — Sys2 u_e 恢复场
- `fig_torso.png` — Sys3 躯干电位
- `fig_ecg.png` — 体表前向 ECG
- `fig_mono_activation.png` — 单域激活时间
- `fig_mono_ecg.png` — pseudo-ECG
- `fig_tt06_ap.png` — TP06 单细胞动作电位
- `fig_pipeline.png` — 三系统求解流水线
- `fig_verify_fields.png` — 场验证(串/并行一致)
- `fig_verify_time.png` — 时间步验证
- `fig_roadmap.png` — 方法路线图

## 文档 / 幻灯片

- `slides_sim_fakegeo.pdf`
- `slides_sim_fakegeo.tex`
- `slides_sim_fakegeo_en.pdf`
- `slides_sim_fakegeo_en.tex`
- `REPORT_T2_cardiac_zh.md`
- `REPORT_T3_xsys_zh.md`
- `REPORT_SUMMARY_zh.md`
- `NOTES_schwarz_variants_zh.md`
