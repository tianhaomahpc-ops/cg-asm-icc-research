// build_ppt.js -- ASM vs sASM unstructured-P1-FEM study deck.
// run: NODE_PATH=$(npm root -g) node build_ppt.js
const pptxgen = require("pptxgenjs");
const sizeOf = (() => { try { return require("image-size"); } catch (e) { return null; } })();
const fs = require("fs");

const P = new pptxgen();
P.layout = "LAYOUT_WIDE";          // 13.3 x 7.5
const W = 13.333, H = 7.5;
P.author = "Tianhao Ma";
P.title = "ASM vs sASM on unstructured P1 FEM";

// ---- palette ----
const BG = "F7F9FC", DARK = "14233B", NAVY = "1B3A5B", TEAL = "117A8B",
      TEALD = "0E4D64", AMBER = "E8A13A", TEXT = "1B2A3A", MUTED = "5B6B7B",
      CARD = "FFFFFF", LINE = "DCE4EC", ASM = "1F4E79", SASM = "C8641B", WHITE = "FFFFFF";
const HF = "Georgia", BF = "Calibri", MF = "Consolas";
const RES = "results/";

function imgSize(p) {
  // fixed known sizes (px) to avoid extra deps
  const m = { "fig_iter_2d.png": [1430,1040], "fig_iter_3d.png":[1430,1040],
    "fig_kappa_2d.png":[1430,1040], "fig_dim_casea.png":[1430,559],
    "fig_dim_casec.png":[1430,559], "fig_bc_effect.png":[1430,559],
    "fig_aniso_fullD.png":[1430,559], "fig_aniso_mixed.png":[1430,559],
    "fig_timing_2d.png":[1416,543], "fig_spectrum.png":[2080,494],
    "fig_mesh_partition.png":[1937,700] };
  const k = p.split("/").pop();
  return m[k] || [1430,559];
}
// contain-fit an image into a box, return {x,y,w,h} centered
function fit(p, bx, by, bw, bh) {
  const [pw, ph] = imgSize(p); const r = pw/ph; let w = bw, h = bw/r;
  if (h > bh) { h = bh; w = bh*r; }
  return { path: RES+p, x: bx+(bw-w)/2, y: by+(bh-h)/2, w, h };
}
function footer(s, n) {
  s.addShape(P.shapes.RECTANGLE, { x:0, y:H-0.32, w:W, h:0.32, fill:{color:BG}, line:{type:"none"} });
  s.addText("ASM vs sASM · 非结构 P1 有限元 · PETSc 3.24", { x:0.5, y:H-0.34, w:8, h:0.3, fontFace:BF, fontSize:9, color:MUTED, align:"left", valign:"middle", margin:0 });
  s.addText(String(n), { x:W-0.9, y:H-0.34, w:0.5, h:0.3, fontFace:BF, fontSize:9, color:MUTED, align:"right", valign:"middle", margin:0 });
}
let PAGE = 0;
// content-slide scaffold: light bg, left accent, title, returns slide
function content(title, kicker) {
  PAGE++;
  const s = P.addSlide(); s.background = { color: BG };
  s.addShape(P.shapes.RECTANGLE, { x:0, y:0, w:0.18, h:H, fill:{color:TEAL}, line:{type:"none"} });
  if (kicker) s.addText(kicker.toUpperCase(), { x:0.55, y:0.34, w:11, h:0.3, fontFace:BF, fontSize:11, color:TEAL, bold:true, charSpacing:2, margin:0 });
  s.addText(title, { x:0.52, y:0.62, w:12.4, h:0.7, fontFace:HF, fontSize:27, bold:true, color:DARK, margin:0 });
  footer(s, PAGE);
  return s;
}
function card(s, x, y, w, h, fill) {
  s.addShape(P.shapes.RECTANGLE, { x, y, w, h, fill:{color:fill||CARD}, line:{color:LINE, width:1},
    shadow:{type:"outer", color:"9AA9B8", blur:7, offset:2, angle:135, opacity:0.22} });
}

// ===================================================================== 1 TITLE
{
  PAGE++; const s = P.addSlide(); s.background = { color: DARK };
  s.addShape(P.shapes.RECTANGLE, { x:0, y:0, w:W, h:0.22, fill:{color:TEAL}, line:{type:"none"} });
  s.addShape(P.shapes.RECTANGLE, { x:0, y:H-0.22, w:W, h:0.22, fill:{color:AMBER}, line:{type:"none"} });
  s.addText("重叠 Schwarz 预条件：ASM（未加权）对比 sASM（加权）", { x:0.9, y:1.7, w:11.5, h:1.0, fontFace:HF, fontSize:33, bold:true, color:WHITE, margin:0 });
  s.addText("非结构 P1 有限元上的维度 · 边界条件 · 各向异性系统研究", { x:0.9, y:2.75, w:11.5, h:0.6, fontFace:BF, fontSize:18, color:"CADCFC", margin:0 });
  // case chips
  const chips = [["a","全Dirichlet · 各向同性"],["b","全Dirichlet · 各向异性"],
                 ["c","Dirichlet+Neumann · 各向同性"],["d","Dirichlet+Neumann · 各向异性"]];
  let cx = 0.9;
  chips.forEach(([k,t])=>{ const wch = 2.85;
    s.addShape(P.shapes.RECTANGLE, { x:cx, y:3.95, w:wch, h:1.0, fill:{color:NAVY}, line:{color:TEAL,width:1} });
    s.addText("Case "+k, { x:cx, y:4.08, w:wch, h:0.4, fontFace:HF, fontSize:17, bold:true, color:AMBER, align:"center", margin:0 });
    s.addText(t, { x:cx+0.1, y:4.46, w:wch-0.2, h:0.45, fontFace:BF, fontSize:10.5, color:"CADCFC", align:"center", valign:"top", margin:0 });
    cx += wch+0.2; });
  s.addText("CG + PCASM + ICC(0) · PETSc 3.24 · 2026-06-18", { x:0.9, y:5.35, w:11, h:0.4, fontFace:MF, fontSize:12, color:"8FA6C0", margin:0 });
}

// ===================================================================== 2 BACKGROUND
{
  const s = content("背景：重叠为什么反而让迭代变多？", "问题");
  const bullets = [
    [{text:"经典印象：", options:{bold:true,color:TEAL}}, {text:"重叠 ↑ → 条件数 ↓ → 迭代 ↓（κ ≤ C(1+H/δ)）。"}],
    [{text:"反常：", options:{bold:true,color:SASM}}, {text:"未加权 BASIC ASM + ICC 不精确局部解 + CG 时，重叠 ↑ → 迭代 ↑（源自 cardioid Sys2/Sys3）。"}],
    [{text:"为何少见：", options:{bold:true,color:TEAL}}, {text:"主流默认 RAS（非对称、配 GMRES，自带单位分解不过计数）；要 CG 对称就得用 BASIC，正好踩坑。"}],
    [{text:"本研究：", options:{bold:true,color:TEAL}}, {text:"在 4 个 case 上系统比较 ASM 与 sASM 的迭代、时间与可解释量（κ、谱、N̂、ω、Dirichlet 信息传递）。"}],
  ];
  s.addText(bullets.map((b,i)=>({text:b, options:{bullet:{indent:14}, breakLine:true, paraSpaceAfter:10}})).flatMap(x=>x.text.map((t,j)=>({...t, options:{...t.options, ...(j===0?x.options:{}), breakLine:j===x.text.length-1}}))),
    { x:0.6, y:1.7, w:7.3, h:4.6, fontFace:BF, fontSize:15.5, color:TEXT, valign:"top" });
  // right: mechanism cards
  card(s, 8.3, 1.8, 4.5, 1.85);
  s.addText([{text:"ASM (BASIC)\n", options:{bold:true,fontSize:16,color:ASM,breakLine:true}},
             {text:"M⁻¹ = Σᵢ Rᵢᵀ Aᵢ⁻¹ Rᵢ", options:{fontFace:MF,fontSize:13,breakLine:true}},
             {text:"重叠区被重复累加 → 过计数 N̂", options:{fontSize:11.5,color:MUTED}}],
    { x:8.5, y:1.95, w:4.1, h:1.55, valign:"top", lineSpacingMultiple:1.1 });
  card(s, 8.3, 3.85, 4.5, 2.0);
  s.addText([{text:"sASM (加权)\n", options:{bold:true,fontSize:16,color:SASM,breakLine:true}},
             {text:"M⁻¹ = D⁻¹ᐟ² (Σᵢ Rᵢᵀ Aᵢ⁻¹ Rᵢ) D⁻¹ᐟ²", options:{fontFace:MF,fontSize:12.5,breakLine:true}},
             {text:"D = diag(重数 mₖ) → 去掉过计数 N̂，保持对称（CG 可用）", options:{fontSize:11.5,color:MUTED}}],
    { x:8.5, y:4.0, w:4.1, h:1.7, valign:"top", lineSpacingMultiple:1.1 });
}

// ===================================================================== 3 METHOD
{
  const s = content("方法：方程、离散、子域、求解器", "方法");
  card(s, 0.6, 1.65, 5.1, 2.2);
  s.addText([
    {text:"控制方程\n", options:{bold:true,color:TEAL,fontSize:14,breakLine:true}},
    {text:"−∇·(A ∇u) = f   on  Ω=[0,1]ᵈ\n", options:{fontFace:MF,fontSize:14,breakLine:true,color:TEXT}},
    {text:"A：扩散张量（各向同/异性）\n", options:{fontSize:12,color:MUTED,breakLine:true}},
    {text:"离散：P1 非结构单纯形（Delaunay 抖动网格）", options:{fontSize:12.5,color:TEXT}},
  ], { x:0.8, y:1.8, w:4.7, h:1.95, valign:"top", lineSpacingMultiple:1.15 });
  card(s, 0.6, 4.0, 5.1, 2.9);
  s.addText([
    {text:"求解与测量\n", options:{bold:true,color:TEAL,fontSize:14,breakLine:true}},
    {text:"• 子域：S^d 几何盒子 + METIS k-way（两种）\n", options:{fontSize:12.5,breakLine:true}},
    {text:"• 局部解：ICC(0)（不精确）/ Cholesky（精确）\n", options:{fontSize:12.5,breakLine:true}},
    {text:"• Krylov：CG（对称）；重叠 O = 0…4\n", options:{fontSize:12.5,breakLine:true}},
    {text:"• 测量：迭代数、setup/solve 时间、κ、谱分布、\n   N̂（最大重数）、ω=maxᵢλmax(Mᵢ⁻¹Aᵢ)", options:{fontSize:12.5}},
  ], { x:0.8, y:4.15, w:4.7, h:2.65, valign:"top", lineSpacingMultiple:1.12 });
  const im = fit("fig_mesh_partition.png", 6.0, 1.9, 7.0, 4.6);
  s.addImage(im);
  s.addText("2D 示例：非结构 P1 网格与两种子域划分（几何盒子 / METIS）", { x:6.0, y:6.5, w:7.0, h:0.35, fontFace:BF, fontSize:11, italic:true, color:MUTED, align:"center", margin:0 });
}

// ===================================================================== 4 CASE DEFS
{
  const s = content("四个 Case 的定义", "实验设计");
  const hdr = (t)=>({text:t, options:{fill:{color:NAVY}, color:WHITE, bold:true, fontFace:BF, fontSize:13, align:"center", valign:"middle"}});
  const cell = (t,c)=>({text:t, options:{fontFace:BF, fontSize:12.5, color:c||TEXT, align:"left", valign:"middle", margin:4}});
  const rows = [
    [hdr("Case"), hdr("系数 A"), hdr("边界条件"), hdr("维度")],
    [cell("a",ASM), cell("各向同性  A = I"), cell("全 Dirichlet  (u=0 整个 ∂Ω)"), cell("1D / 2D / 3D")],
    [cell("b",ASM), cell("各向异性  A=diag(σL,σT,σT)\n2D:diag(10,1) · 3D:Niederer σL/σT=7.58 (纤维∥x)"), cell("全 Dirichlet"), cell("2D / 3D")],
    [cell("c",ASM), cell("各向同性  A = I"), cell("混合：x₀=0 面 Dirichlet，其余 ∂u/∂n=0"), cell("1D / 2D / 3D")],
    [cell("d",ASM), cell("各向异性（同 b）"), cell("混合（同 c）"), cell("2D / 3D")],
  ];
  s.addTable(rows, { x:0.6, y:1.7, w:9.4, colW:[0.7,3.9,3.1,1.7], rowH:[0.4,0.55,0.95,0.6,0.55],
    border:{pt:0.5,color:LINE}, fill:{color:CARD}, valign:"middle" });
  card(s, 10.3, 1.7, 2.55, 3.6, "0E2236");
  s.addText([
    {text:"对比维度\n\n", options:{bold:true,color:AMBER,fontSize:14,breakLine:true}},
    {text:"b vs a → 各向异性的影响\n\n", options:{color:"CADCFC",fontSize:12.5,breakLine:true}},
    {text:"c vs a → 边界条件的影响\n\n", options:{color:"CADCFC",fontSize:12.5,breakLine:true}},
    {text:"d vs c → 各向异性 × 混合BC", options:{color:"CADCFC",fontSize:12.5}},
  ], { x:10.5, y:1.9, w:2.2, h:3.2, valign:"top", lineSpacingMultiple:1.05 });
  s.addText([
    {text:"预条件子：  ", options:{color:MUTED,fontSize:12}},
    {text:"ASM  M⁻¹=ΣRᵢᵀAᵢ⁻¹Rᵢ", options:{fontFace:MF,color:ASM,fontSize:13,bold:true}},
    {text:"     vs     ", options:{color:MUTED,fontSize:12}},
    {text:"sASM  M⁻¹=D⁻¹ᐟ²(·)D⁻¹ᐟ²", options:{fontFace:MF,color:SASM,fontSize:13,bold:true}},
  ], { x:0.6, y:6.05, w:9.4, h:0.5, valign:"middle", margin:0 });
}

// helper for a "result" slide: big image + analysis card
function resultSlide(title, kicker, img, imgCap, analysis, opts) {
  opts = opts || {};
  const s = content(title, kicker);
  const iw = opts.iw || 8.2, ih = opts.ih || 4.85;
  const im = fit(img, 0.5, 1.6, iw, ih); s.addImage(im);
  s.addText(imgCap, { x:0.5, y:1.6+ih, w:iw, h:0.35, fontFace:BF, fontSize:11, italic:true, color:MUTED, align:"center", margin:0 });
  const ax = 0.5+iw+0.15;
  card(s, ax, 1.65, W-ax-0.45, 4.9);
  s.addText(analysis.map((b,i)=>({ text:b.t, options:{ bullet:{indent:13}, breakLine:true, paraSpaceAfter:8,
      fontSize:b.s||13, bold:!!b.b, color:b.c||TEXT } })),
    { x:ax+0.18, y:1.82, w:W-ax-0.8, h:4.6, fontFace:BF, valign:"top" });
  return s;
}

// ===================================================================== 5 OVERVIEW 2D
resultSlide("总览：ASM vs sASM 迭代次数（2D）", "结果 · 迭代",
  "fig_iter_2d.png", "图：2D 各 case，CG 迭代 vs 重叠；实线=ASM，虚线=sASM；蓝=盒子, 橙=METIS。",
  [
    {t:"本图展示：四个 case 的迭代-重叠曲线，对比 ASM 与 sASM。", b:true, c:TEAL},
    {t:"ASM：重叠↑ 迭代↑（反常）—— a 79→104, c 119→158。"},
    {t:"sASM：单调下降并显著更低 —— a 79→64, c 119→95。"},
    {t:"混合BC(c) 比全Dirichlet(a) 反常更强。"},
    {t:"box 与 METIS 划分趋势一致。", c:MUTED, s:12},
  ], { iw:8.4, ih:4.95 });

// ===================================================================== 6 3D
resultSlide("ASM vs sASM（3D）：最严重的组合", "结果 · 3D",
  "fig_iter_3d.png", "图：3D 各 case，CG 迭代 vs 重叠；实线=ASM，虚线=sASM。",
  [
    {t:"本图展示：3D（N≈8000，~49k 四面体）的迭代-重叠曲线。", b:true, c:TEAL},
    {t:"Case d（混合BC+各向异性）最坏：ASM 203→413（翻倍），sASM 203→143。", c:SASM},
    {t:"sASM 在所有 3D case 全面更优。"},
    {t:"3D 几何过计数 N̂ 最大可达 ~20（盒子角点）。", s:12},
    {t:"ω 在非结构 3D P1 升到 2–5（ICC 在不规则网格更差）。", c:MUTED, s:12},
  ], { iw:8.4, ih:4.95 });

// ===================================================================== 7 KAPPA
resultSlide("条件数 κ(M⁻¹A)（2D）", "结果 · 条件数",
  "fig_kappa_2d.png", "图：2D 各 case，κ(M⁻¹A) vs 重叠（对数轴）；实线=ASM，虚线=sASM。",
  [
    {t:"本图展示：预条件后条件数随重叠的变化。", b:true, c:TEAL},
    {t:"ASM：κ 随重叠上升或不降（λmax 被 N̂ 抬高）。"},
    {t:"sASM：κ 明显更低（去掉 N̂ 过计数）。"},
    {t:"迭代数与 κ 的趋势一致，互相印证。", c:MUTED, s:12},
  ], { iw:8.4, ih:4.95 });

// ===================================================================== 8 DIMENSION
resultSlide("维度效应：1D 免疫，2D/3D 出现反常", "可解释性 · 维度",
  "fig_dim_casea.png", "图：Case a（全Dirichlet 各向同性，盒子划分）1D/2D/3D 的迭代与 κ。",
  [
    {t:"本图展示：固定 case a，比较 1D/2D/3D。", b:true, c:TEAL},
    {t:"1D：ICC(0) 在三对角阵上精确 → ω=1 → 无反常；ASM 迭代恒为 17。"},
    {t:"sASM 在 1D 反而更差（17→34）：无过计数可去，加权只扰动。", c:SASM},
    {t:"2D/3D：ω>1，过计数 N̂=2ᵈ 起作用 → 反常出现，sASM 修复。"},
    {t:"维度放大器是几何 N̂=m_axis^d，不是 ω（ω 饱和）。", c:MUTED, s:12},
  ], { iw:8.4, ih:4.3 });

// ===================================================================== 9 BC EFFECT
resultSlide("边界条件效应：Dirichlet 信息传递", "可解释性 · 边界条件",
  "fig_bc_effect.png", "图：2D，全Dirichlet(a) vs 混合BC(c)：迭代 与 λmin vs 重叠。",
  [
    {t:"本图展示：边界条件如何改变反常与 λmin。", b:true, c:TEAL},
    {t:"全Dirichlet(a)：整个边界被钉住，信息从四面传入 → λmin 较大，反常较轻。"},
    {t:"混合(c)：仅一面 Dirichlet，Neumann 方向使 λmin 更小、信息传播更慢 → 反常更强。", c:SASM},
    {t:"重叠↑ 提升 λmin（加速边界信息传递），但 ASM 的 N̂ 过计数抵消了收益。", s:12, c:MUTED},
  ], { iw:8.4, ih:4.6 });

// ===================================================================== 10 ANISOTROPY
{
  const s = content("各向异性效应：抬高 ω 与迭代，并与混合BC 叠加", "可解释性 · 各向异性");
  const a = fit("fig_aniso_fullD.png", 0.4, 1.55, 6.4, 2.4); s.addImage(a);
  const b = fit("fig_aniso_mixed.png", 0.4, 4.1, 6.4, 1.95); s.addImage(b);
  s.addText("上：全Dirichlet  a(各向同) vs b(各向异)　·　下：混合BC  c vs d", { x:0.4, y:6.15, w:6.4, h:0.3, fontFace:BF, fontSize:10.5, italic:true, color:MUTED, align:"center", margin:0 });
  card(s, 7.0, 1.7, W-7.0-0.45, 5.0);
  s.addText([
    {t:"本图展示：在全Dirichlet 与混合BC 下，各向异性对迭代/λmin 的影响。", b:true, c:TEAL},
    {t:"各向异性（2D 10×、3D Niederer 7.58）抬高局部 ICC 放大因子 ω 与迭代数。"},
    {t:"几何过计数 N̂ 与系数无关（纯几何），不被各向异性改变。"},
    {t:"最坏是 case d-3D（各向异性 × 混合BC）：ASM 迭代翻倍。", c:SASM},
    {t:"sASM 去掉 N̂ 修复重叠趋势；但 sASM 几何加权对各向异性「盲」——强各向异性/高对比下绝对 κ 仍高，需谱粗空间(GenEO)。", s:12, c:MUTED},
  ].map(x=>({text:x.t, options:{bullet:{indent:13}, breakLine:true, paraSpaceAfter:9, fontSize:x.s||13, bold:!!x.b, color:x.c||TEXT}})),
    { x:7.2, y:1.86, w:W-7.0-0.85, h:4.7, fontFace:BF, valign:"top" });
}

// ===================================================================== 11 SPECTRUM
resultSlide("谱分布（CG Ritz 值）", "可解释性 · 谱",
  "fig_spectrum.png", "图：代表算例 O=2 的 M⁻¹A Ritz 特征值分布直方图；蓝=ASM, 红=sASM。",
  [
    {t:"本图展示：预条件算子的谱分布。", b:true, c:TEAL},
    {t:"ASM：谱向右拖出大特征值簇（N̂ 过计数把 λmax 推高）。"},
    {t:"sASM：大端被压回 ≈ ω，谱更集中 → CG 收敛更快。", c:SASM},
    {t:"谱宽 = 条件数，直接对应迭代数差异。", s:12, c:MUTED},
  ], { iw:8.6, ih:3.2 });

// ===================================================================== 12 TIMING
resultSlide("计算时间：sASM 每步略贵，但更快收敛", "结果 · 时间",
  "fig_timing_2d.png", "图：2D，solve 与 setup 墙钟时间 vs 重叠；实线=ASM, 虚线=sASM。",
  [
    {t:"本图展示：ASM 与 sASM 的求解/装配时间。", b:true, c:TEAL},
    {t:"sASM 每次迭代仅多 2 次向量缩放（D⁻¹ᐟ²）——开销极小。"},
    {t:"迭代数更少 → sASM 总求解时间通常更短。", c:SASM},
    {t:"setup 两者相当（sASM 复用同一 BASIC 内核 + 一次重数计算）。", s:12, c:MUTED},
    {t:"注：本机串行计时，为相对比较；HPC 并行规律一致。", s:11, c:MUTED},
  ], { iw:8.4, ih:4.6 });

// ===================================================================== 13 CONCLUSIONS
{
  PAGE++; const s = P.addSlide(); s.background = { color: DARK };
  s.addShape(P.shapes.RECTANGLE, { x:0, y:0, w:0.18, h:H, fill:{color:AMBER}, line:{type:"none"} });
  s.addText("结论", { x:0.6, y:0.5, w:11, h:0.7, fontFace:HF, fontSize:30, bold:true, color:WHITE, margin:0 });
  const cc = [
    [{text:"sASM 在 2D/3D 全面优于 ASM：", options:{bold:true,color:AMBER}}, {text:"去掉几何过计数 N̂，迭代单调下降、κ 更低、总时间更短；保持对称（CG 可用）。", options:{color:"E6EEF8"}}],
    [{text:"1D 是例外：", options:{bold:true,color:AMBER}}, {text:"ICC(0) 精确(ω=1)、无过计数 → 无反常；此时 sASM 反而略差。", options:{color:"E6EEF8"}}],
    [{text:"最严重的组合：", options:{bold:true,color:AMBER}}, {text:"3D + 混合边界 + 各向异性（case d-3D）ASM 迭代随重叠翻倍。", options:{color:"E6EEF8"}}],
    [{text:"边界条件：", options:{bold:true,color:AMBER}}, {text:"全Dirichlet 信息从四面传入 → λmin 大、反常轻；混合BC 反之。", options:{color:"E6EEF8"}}],
    [{text:"机制：", options:{bold:true,color:AMBER}}, {text:"λmax(ASM) ≈ ω·N̂；N̂=几何过计数(维度放大器), ω=不精确ICC放大(开关)。", options:{color:"E6EEF8"}}],
    [{text:"边界：", options:{bold:true,color:AMBER}}, {text:"sASM 几何加权对各向异性/高对比「盲」——这些情形需谱粗空间 (GenEO)。", options:{color:"E6EEF8"}}],
  ];
  s.addText(cc.flatMap(r=>r.map((t,j)=>({...t, options:{...t.options, bullet:{indent:15}, breakLine:j===r.length-1, paraSpaceAfter:j===r.length-1?12:0, fontSize:15}}))),
    { x:0.6, y:1.4, w:12.2, h:5.6, fontFace:BF, valign:"top" });
  footer(s, PAGE);
}

// ===================================================================== 14 REPRO
{
  const s = content("复现与工具", "附录");
  card(s, 0.6, 1.7, 7.4, 4.9);
  s.addText([
    {text:"流程（PETSc 3.24）\n", options:{bold:true,color:TEAL,fontSize:14,breakLine:true}},
    {text:"python3 fem_build.py <a|b|c|d> <dim> <n> <nsub>\n", options:{fontFace:MF,fontSize:12,breakLine:true}},
    {text:"   非结构 P1 网格(Delaunay)+装配+两种分区 → PETSc 二进制\n", options:{fontSize:11,color:MUTED,breakLine:true}},
    {text:"mpicc schwarz_fem.c -lpetsc -o schwarz_fem\n", options:{fontFace:MF,fontSize:12,breakLine:true}},
    {text:"bash run_fem.sh        # 全 case × {box,METIS} × {ASM,sASM} × O\n", options:{fontFace:MF,fontSize:12,breakLine:true}},
    {text:"python3 parse_fem.py   # → results/*.csv + 图\n\n", options:{fontFace:MF,fontSize:12,breakLine:true}},
    {text:"测量量：iter, t_setup, t_solve, κ, 谱(Ritz), N̂, ω", options:{fontSize:12.5,color:TEXT}},
  ], { x:0.8, y:1.85, w:7.0, h:4.6, valign:"top", lineSpacingMultiple:1.12 });
  card(s, 8.2, 1.7, W-8.2-0.45, 4.9, "0E2236");
  s.addText([
    {text:"要点\n\n", options:{bold:true,color:AMBER,fontSize:14,breakLine:true}},
    {text:"• 串行 + 显式 S^d/METIS 子域，独立于进程数\n\n", options:{color:"CADCFC",fontSize:12.5,breakLine:true}},
    {text:"• 工具为 PETSc 3.24 API，可直接搬到 HPC spack PETSc 3.24 跑大规模\n\n", options:{color:"CADCFC",fontSize:12.5,breakLine:true}},
    {text:"• 高对比/各向异性局部 ICC 用 positive_definite shift\n\n", options:{color:"CADCFC",fontSize:12.5,breakLine:true}},
    {text:"• 下一步：在 case d 边界处加 GenEO 谱粗空间", options:{color:"CADCFC",fontSize:12.5}},
  ], { x:8.4, y:1.9, w:W-8.2-0.85, h:4.5, valign:"top" });
}

P.writeFile({ fileName: "ASM_vs_sASM_FEM_study.pptx" }).then(f => console.log("wrote", f));
