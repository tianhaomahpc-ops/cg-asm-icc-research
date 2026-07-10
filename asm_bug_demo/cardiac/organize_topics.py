#!/usr/bin/env python3
"""organize_topics.py -- group this session's figures/docs/slides into topic
folders under topics/ WITHOUT moving originals (LaTeX .tex and plot .py scripts
reference figures by relative path in this dir, so originals stay put). Each topic
folder gets COPIES of its figures + relevant docs/slides + a README index.
Re-runnable: it rebuilds topics/ from scratch each time.
"""
import os, shutil, textwrap

HERE = os.path.dirname(os.path.abspath(__file__))
TOP  = os.path.join(HERE, "topics")

# topic slug -> (title, blurb, [ (figure, caption) ... ], [docs/slides ...])
TOPICS = {
"01_fakegeo_sASM_result": (
  "假几何+真实心脏参数:ASM vs sASM 结果(汇报用)",
  "cardiac-sim-fakegeo 分支的主结果:三系统耦合前向 ECG,真几何 FEM,ASM(BASIC) vs "
  "sASM 的迭代数与时间。**汇报 slides 在此。**",
  [("fig_mesh.png","conforming 心脏嵌躯干网格"),
   ("fig_meshview.png","网格切面"),
   ("fig_geom.png","几何/子域分区"),
   ("fig_vm.png","Sys1 单域 Vm 场"),
   ("fig_ue.png","Sys2 u_e 恢复场"),
   ("fig_torso.png","Sys3 躯干电位"),
   ("fig_ecg.png","体表前向 ECG"),
   ("fig_mono_activation.png","单域激活时间"),
   ("fig_mono_ecg.png","pseudo-ECG"),
   ("fig_tt06_ap.png","TP06 单细胞动作电位"),
   ("fig_pipeline.png","三系统求解流水线"),
   ("fig_verify_fields.png","场验证(串/并行一致)"),
   ("fig_verify_time.png","时间步验证"),
   ("fig_roadmap.png","方法路线图")],
  ["slides_sim_fakegeo.pdf","slides_sim_fakegeo.tex",
   "slides_sim_fakegeo_en.pdf","slides_sim_fakegeo_en.tex",
   "REPORT_T2_cardiac_zh.md","REPORT_T3_xsys_zh.md","REPORT_SUMMARY_zh.md",
   "NOTES_schwarz_variants_zh.md"]),

"02_teaching_CG_spectrum": (
  "教学:CG 过程 / 谱 / 模 / 波长(可解释性)",
  "把 CG 求解、特征值、慢模、波长、粗空间从零讲透的全部图与教程。",
  [("fig_cg_process.png","CG 过程本身(2 步,无特征值)"),
   ("fig_bowl_cg.png","解=找碗底;CG vs 最陡下降"),
   ("fig_alpha_and_lambda.png","α 的上限 + 每模折扣 1-αλ"),
   ("fig_lambda_meaning.png","特征值=碗的陡峭度=收敛速度"),
   ("fig_cg_polynomial.png","CG 滤波多项式 p_k(λ)"),
   ("fig_filter_eq.png","p_k(λ) 当均衡器"),
   ("fig_two_views.png","一个例子两条路(特征值/CG)"),
   ("fig_modes_explain.png","模的形状(堆叠)"),
   ("fig_vector_two_views.png","向量两种画法:箭头/剖面"),
   ("fig_profile_meaning.png","剖面=场;快慢定义;W 来源"),
   ("fig_spectrum_coarse.png","谱→谱分布→CG→粗空间(1D)"),
   ("fig_solve_linked.png","共轴热力图:谱/CG/deflated"),
   ("fig_tut_chain.png","弹珠链 n=3:模与逐模衰减"),
   ("fig_tut_spectrum.png","弹珠链 n=100:谱+热力图"),
   ("fig_poisson_modes.png","Poisson 长条:问题/解/模"),
   ("fig_poisson_cg.png","Poisson+CG:谱/实况/粗空间")],
  ["TUTORIAL_chain_zh.md","TUTORIAL_three_systems_zh.md"]),

"03_coarse_space_deflation": (
  "粗空间 / deflation(真实 Sys2)",
  "在真实 Sys2 算子上做粗空间/deflation 的结果:迭代数与残差历史。",
  [("fig_deflation.png","粗空间机制示意"),
   ("fig_deflate_real.png","真实 Sys2:fine/Nicolaides/geometric 迭代"),
   ("fig_deflate_reshist.png","真实 Sys2 残差历史:有无 deflation"),
   ("fig_method_on_systems.png","三系统:谱/残差尾/杠杆条")],
  []),

"04_recycling_fischer": (
  "回收 / Fischer(EP 时间循环)",
  "跨时间步回收把重复求解成本摊掉的全部结果。",
  [("fig_fischer_eploop.png","真实 EP 循环:每步迭代数 cold vs Fischer"),
   ("fig_fischer_demo.png","Fischer 机制:A-正交投影(可复现)"),
   ("fig_fischer_batched.png","大规模改造:打包+CGS2 前后对比")],
  ["fischer_eploop.txt","fischer_before_iters.txt","fischer_after_iters.txt"]),

"05_scaling_3000core": (
  "大规模可扩展性 / 3000 核方案",
  "慢模维数 scaling、实测集群数据、3000 核三层设计。",
  [("fig_slowdim_scaling.png","慢模维数~O(子域数)的 scaling"),
   ("fig_cluster_scaling.png","实测集群 384/768/1536 核扩展性"),
   ("fig_3000core_design.png","3000 核三层求解器设计")],
  ["DESIGN_3000core_zh.md","RESEARCH_beyond_coarse_zh.md"]),
}

def main():
    if os.path.isdir(TOP): shutil.rmtree(TOP)
    os.makedirs(TOP)
    index_lines = ["# 论题索引(topics/)\n",
                   "> 每个文件夹含该论题的图、文档/幻灯片副本,和一份 README。",
                   "> 原文件仍在 cardiac/ 根目录(脚本与 LaTeX 依赖其相对路径),此处为归档副本。\n"]
    for slug,(title,blurb,figs,docs) in TOPICS.items():
        d = os.path.join(TOP, slug); os.makedirs(d)
        rl = [f"# {title}\n", blurb, "\n## 图\n"]
        for fn,cap in figs:
            src = os.path.join(HERE, fn)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(d, fn)); rl.append(f"- `{fn}` — {cap}")
            else:
                rl.append(f"- (缺) `{fn}` — {cap}")
        if docs:
            rl.append("\n## 文档 / 幻灯片\n")
            for fn in docs:
                src = os.path.join(HERE, fn)
                if os.path.exists(src):
                    shutil.copy2(src, os.path.join(d, fn)); rl.append(f"- `{fn}`")
                else:
                    rl.append(f"- (缺) `{fn}`")
        open(os.path.join(d,"README.md"),"w").write("\n".join(rl)+"\n")
        index_lines.append(f"- **[{slug}]({slug}/)** — {title}")
    open(os.path.join(TOP,"INDEX.md"),"w").write("\n".join(index_lines)+"\n")
    # report
    for slug in TOPICS:
        n=len([f for f in os.listdir(os.path.join(TOP,slug))])
        print(f"{slug}: {n} files")
    print(f"\nwrote {TOP}/INDEX.md")

if __name__=="__main__": main()
