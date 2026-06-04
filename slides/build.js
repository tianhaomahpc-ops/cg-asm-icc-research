// build.js — ASM overlap anomaly deck (12 slides, Chinese, focused)
const pptxgen = require("pptxgenjs");

const P = new pptxgen();
P.layout = "LAYOUT_WIDE";          // 13.33 x 7.5 in
P.author = "Tianhao Ma";
P.title  = "ASM overlap anomaly: diagnosis and single-level fix";

// ---- palette (scientific HPC, semantic red=up/bad, green=down/good) ----
const C = {
  navy:   "15233F",   // dark bg
  navy2:  "1E3358",
  ink:    "1A2B4A",   // primary text on light
  muted:  "6B7A99",
  light:  "F5F7FB",   // content bg
  card:   "FFFFFF",
  teal:   "1C7293",   // accent
  ice:    "CADCFC",
  red:    "C0392B",   // anomaly, iter up
  redbg:  "FAEAE8",
  green:  "1E8A5B",   // fix, iter down
  greenbg:"E7F4ED",
  line:   "DCE3F0",
};
const F = "PingFang SC";           // CJK
const MONO = "Menlo";              // ASCII data / formulas
const W = 13.33, H = 7.5, M = 0.7;

// ---------- helpers ----------
const shadow = () => ({ type: "outer", color: "8090B0", blur: 8, offset: 3, angle: 135, opacity: 0.18 });

function footer(s, n, total = 12) {
  s.addText(`${n} / ${total}`, { x: W - 1.4, y: H - 0.5, w: 1.0, h: 0.3,
    fontFace: MONO, fontSize: 10, color: C.muted, align: "right" });
  s.addText("ASM overlap ↑ → iter ↑ :  诊断与单层修复", { x: M, y: H - 0.5, w: 8, h: 0.3,
    fontFace: F, fontSize: 9.5, color: C.muted, align: "left" });
}

// light content slide header: title (left) + optional problem tag (right)
function header(s, title, tag) {
  s.background = { color: C.light };
  s.addShape(P.shapes.RECTANGLE, { x: 0, y: 0, w: 0.18, h: H, fill: { color: C.teal } });
  s.addText(title, { x: M, y: 0.45, w: tag ? 7.7 : 11.9, h: 0.85, fontFace: F, fontSize: 26, bold: true,
    color: C.ink, align: "left", valign: "middle", margin: 0 });
  if (tag) {
    s.addShape(P.shapes.ROUNDED_RECTANGLE, { x: W - 4.55, y: 0.55, w: 3.85, h: 0.62,
      fill: { color: C.navy }, rectRadius: 0.08 });
    s.addText([
      { text: "问题  ", options: { color: C.ice, fontSize: 11 } },
      { text: tag, options: { color: "FFFFFF", fontSize: 12.5, bold: true } },
    ], { x: W - 4.55, y: 0.55, w: 3.85, h: 0.62, fontFace: F, align: "center", valign: "middle", margin: 0 });
  }
}

// a colored "pill" for trend
function trendPill(s, x, y, txt, kind) {
  const col = kind === "up" ? C.red : kind === "down" ? C.green : C.muted;
  const bg  = kind === "up" ? C.redbg : kind === "down" ? C.greenbg : "EEF1F7";
  s.addShape(P.shapes.ROUNDED_RECTANGLE, { x, y, w: 1.5, h: 0.42, fill: { color: bg }, rectRadius: 0.06 });
  s.addText(txt, { x, y, w: 1.5, h: 0.42, fontFace: F, fontSize: 13, bold: true, color: col,
    align: "center", valign: "middle", margin: 0 });
}

// section/title/conclusion dark slide scaffold
function darkBg(s) {
  s.background = { color: C.navy };
  s.addShape(P.shapes.RECTANGLE, { x: 0, y: 0, w: W, h: 0.12, fill: { color: C.teal } });
  s.addShape(P.shapes.RECTANGLE, { x: 0, y: H - 0.12, w: W, h: 0.12, fill: { color: C.teal } });
}

// =========================================================================
// S1 — Title
// =========================================================================
{
  const s = P.addSlide(); darkBg(s);
  s.addText("为什么增大 ASM overlap,CG 迭代反而增加?", {
    x: 1.0, y: 2.0, w: 11.3, h: 1.2, fontFace: F, fontSize: 38, bold: true, color: "FFFFFF",
    align: "left", margin: 0 });
  s.addText("椭圆问题上的诊断与单层修复 ——  保持 CG、不引入两层方法", {
    x: 1.0, y: 3.25, w: 11.3, h: 0.6, fontFace: F, fontSize: 19, color: C.ice, align: "left", margin: 0 });
  // accent rule
  s.addShape(P.shapes.RECTANGLE, { x: 1.02, y: 4.05, w: 2.4, h: 0.05, fill: { color: C.teal } });
  s.addText([
    { text: "MFEM 4.9 + PETSc 3.24", options: { bold: true } },
    { text: "   ·   双实现交叉验证(MFEM / 纯 PETSc)   ·   立方体 P1 有限元", options: {} },
  ], { x: 1.0, y: 4.4, w: 11.3, h: 0.5, fontFace: F, fontSize: 14, color: "C9D4E8", align: "left", margin: 0 });
  s.addText("github.com/tianhaomahpc-ops/cg-asm-icc-research", {
    x: 1.0, y: 6.5, w: 11.3, h: 0.4, fontFace: MONO, fontSize: 12.5, color: "6FB3CE", align: "left", margin: 0 });
}

// =========================================================================
// S2 — phenomenon + core questions
// =========================================================================
{
  const s = P.addSlide(); header(s, "现象:overlap 增大,迭代反而上升");
  // left card: cardioid observed data
  s.addShape(P.shapes.RECTANGLE, { x: M, y: 1.55, w: 5.55, h: 4.7, fill: { color: C.card }, shadow: shadow() });
  s.addText("cardioid 实测(Sys3 / Torso,ICC L=0)", { x: M + 0.3, y: 1.8, w: 5.0, h: 0.4,
    fontFace: F, fontSize: 14, bold: true, color: C.ink, margin: 0 });
  s.addTable([
    [{ text: "overlap", options: hCell() }, { text: "平均迭代次数", options: hCell() }],
    [{ text: "O = 0", options: bCell() }, { text: "115", options: bCell() }],
    [{ text: "O = 1", options: bCell(C.red) }, { text: "200  (撞 max_it)", options: bCell(C.red) }],
    [{ text: "O = 2", options: bCell(C.red) }, { text: "200  (撞 max_it)", options: bCell(C.red) }],
  ], { x: M + 0.3, y: 2.35, w: 4.95, colW: [2.0, 2.95], rowH: 0.55, fontFace: F, fontSize: 14,
       border: { pt: 0.5, color: C.line }, valign: "middle", align: "center" });
  trendPill(s, M + 0.3, 4.55, "↑  上升", "up");
  s.addText("经典 Schwarz 理论:overlap↑ 应当 iter↓。实测相反。", {
    x: M + 0.3, y: 5.25, w: 4.95, h: 0.9, fontFace: F, fontSize: 13.5, color: C.red, italic: true, margin: 0 });

  // right: two core questions
  s.addText("本工作只回答两个问题", { x: 7.0, y: 1.7, w: 5.6, h: 0.5, fontFace: F, fontSize: 16, bold: true, color: C.teal, margin: 0 });
  qBlock(s, 7.0, 2.35, "①", "为什么?", "增大 overlap 为何使 CG 迭代上升,是不是代码 bug?");
  qBlock(s, 7.0, 3.75, "②", "怎么修?", "在保持 CG、不引入两层方法的前提下,如何修复?");
  // pre-conclusion strip
  s.addShape(P.shapes.RECTANGLE, { x: 7.0, y: 5.45, w: 5.6, h: 0.95, fill: { color: C.navy } });
  s.addText([
    { text: "预告:", options: { bold: true, color: C.ice } },
    { text: "不是 bug;病根 = 子域解不精确 × 重叠重复计数;修复 = sASM(+Chebyshev),保 CG、单层。", options: { color: "FFFFFF" } },
  ], { x: 7.2, y: 5.45, w: 5.2, h: 0.95, fontFace: F, fontSize: 12.5, align: "left", valign: "middle", margin: 0 });
  footer(s, 2);
}

// =========================================================================
// S3 — abstraction + framework
// =========================================================================
{
  const s = P.addSlide(); header(s, "抽象:统一到一个干净的模型问题");
  // main test problem card
  s.addShape(P.shapes.RECTANGLE, { x: M, y: 1.55, w: 7.1, h: 3.0, fill: { color: C.card }, shadow: shadow() });
  s.addShape(P.shapes.RECTANGLE, { x: M, y: 1.55, w: 0.1, h: 3.0, fill: { color: C.teal } });
  s.addText("主测问题(贯穿全部实验链):Sys3 / Laplace", { x: M + 0.35, y: 1.75, w: 6.5, h: 0.45,
    fontFace: F, fontSize: 16, bold: true, color: C.ink, margin: 0 });
  s.addText([
    { text: "− Δu = 1", options: { bold: true, breakLine: true } },
    { text: "u = 0          于 x = 0 面          (Dirichlet)", options: { breakLine: true } },
    { text: "∂u/∂n = 0      于其余 5 面          (Neumann)", options: {} },
  ], { x: M + 0.45, y: 2.35, w: 6.4, h: 1.4, fontFace: MONO, fontSize: 15, color: C.ink, lineSpacingMultiple: 1.25, margin: 0 });
  s.addText("立方体 [0,1]³ · P1 四面体 · (nx+1)³ 顶点自由度", { x: M + 0.45, y: 3.9, w: 6.4, h: 0.5,
    fontFace: F, fontSize: 13, color: C.muted, margin: 0 });

  // framework chips on right
  s.addText("统一求解框架", { x: 8.1, y: 1.65, w: 4.5, h: 0.4, fontFace: F, fontSize: 15, bold: true, color: C.teal, margin: 0 });
  chip(s, 8.1, 2.15, "CG", "preconditioned norm, rtol 1e-6");
  chip(s, 8.1, 2.86, "ASM(BASIC)", "overlap O = 0 / 1 / 2");
  chip(s, 8.1, 3.57, "子域 ICC(L)", "fill level L = 0 / 1 / 2");
  chip(s, 8.1, 4.28, "两套独立实现", "MFEM 装配 / 纯 PETSc");

  // universality bar
  s.addShape(P.shapes.RECTANGLE, { x: M, y: 5.15, w: 11.93, h: 1.2, fill: { color: "ECF1FA" } });
  s.addText([
    { text: "另备两个问题作普遍性检验(第四部分):  ", options: { bold: true, color: C.ink } },
    { text: "pure Neumann(奇异椭圆)", options: { color: C.teal, bold: true } },
    { text: "   与   ", options: { color: C.muted } },
    { text: "reaction-diffusion(质量主导)", options: { color: C.teal, bold: true } },
    { text: "  —— 验证同现象、同修复。", options: { color: C.ink } },
  ], { x: M + 0.3, y: 5.15, w: 11.3, h: 1.2, fontFace: F, fontSize: 13.5, align: "left", valign: "middle", margin: 0 });
  footer(s, 3);
}

// =========================================================================
// S4 — diagnosis: two factors
// =========================================================================
{
  const s = P.addSlide(); header(s, "诊断:迭代数受条件数支配,κ 含两个病灶");
  s.addText([
    { text: "n", options: { italic: true } },
    { text: "iter", options: { fontSize: 13, italic: true } },
    { text: "  ~  √κ ,        κ = κ( M⁻¹A )", options: {} },
  ], { x: M, y: 1.6, w: 11.9, h: 0.6, fontFace: MONO, fontSize: 20, bold: true, color: C.ink, align: "center", margin: 0 });

  // the bound with three factors as three cards (κ≲ + 3 cards, right-cleared)
  s.addText("κ  ≲", { x: 0.85, y: 2.55, w: 1.25, h: 1.9, fontFace: MONO, fontSize: 24, bold: true, color: C.ink, align: "center", valign: "middle", margin: 0 });
  factorCard(s, 2.3,  2.55, C.teal, "C₀²(δ)", "稳定分解", "overlap δ↑ 时 ↓(有利)");
  s.addText("×", { x: 5.5,  y: 2.55, w: 0.4, h: 1.9, fontFace: MONO, fontSize: 22, color: C.muted, align: "center", valign: "middle", margin: 0 });
  factorCard(s, 5.95, 2.55, C.red,  "Nc", "病灶 B:重叠重复计数", "BASIC 把重叠 DOF 按 mₖ 累加");
  s.addText("×", { x: 9.15, y: 2.55, w: 0.4, h: 1.9, fontFace: MONO, fontSize: 22, color: C.muted, align: "center", valign: "middle", margin: 0 });
  factorCard(s, 9.6,  2.55, C.red,  "ωmax/ωmin", "病灶 A:局部解不精确", "ICC(L) ≠ 精确块逆,=1 当精确");

  // proposition
  s.addShape(P.shapes.RECTANGLE, { x: M, y: 4.7, w: 11.93, h: 1.5, fill: { color: C.navy } });
  s.addText([
    { text: "核心命题   ", options: { bold: true, color: C.ice, fontSize: 16 } },
    { text: "当病灶 A 与 B 同时存在,δ 增大使 (Nc · ω) 的放大盖过 C₀² 的下降  ⇒  κ ↑  ⇒  迭代 ↑。", options: { color: "FFFFFF", fontSize: 15 } },
  ], { x: M + 0.35, y: 4.7, w: 11.2, h: 0.95, fontFace: F, align: "left", valign: "middle", margin: 0 });
  s.addText("下面用实验逐一拆解:拔掉任一因子,趋势即翻回下降。", {
    x: M + 0.35, y: 5.55, w: 11.2, h: 0.5, fontFace: F, fontSize: 13, color: C.ice, italic: true, margin: 0 });
  footer(s, 4);
}

// =========================================================================
// S5 — experiment 1: reproduce + rule out implementation artefact
// =========================================================================
{
  const s = P.addSlide(); header(s, "实验 1:复现 + 排除「实现/装配产物」", "Sys3 / Laplace");
  s.addText("目的:先确认这不是 MFEM / 矩阵转换的人为产物。", {
    x: M, y: 1.5, w: 11.9, h: 0.5, fontFace: F, fontSize: 15, color: C.ink, margin: 0 });

  bigStat(s, M, 2.2, "27 / 27", "两独立实现在同一 METIS 矩阵上\n迭代数逐位一致", C.green);
  bigStat(s, 5.05, 2.2, "逐位相同", "-ksp_monitor 残差历史\n每一步每一位都相同", C.teal);
  // method note card
  s.addShape(P.shapes.RECTANGLE, { x: 9.1, y: 2.2, w: 3.53, h: 3.0, fill: { color: C.card }, shadow: shadow() });
  s.addText("三层对齐", { x: 9.35, y: 2.4, w: 3.1, h: 0.4, fontFace: F, fontSize: 14, bold: true, color: C.ink, margin: 0 });
  s.addText([
    { text: "矩阵零容差过滤", options: { bullet: true, breakLine: true } },
    { text: "共享 METIS 分区", options: { bullet: true, breakLine: true } },
    { text: "MatLoad 行布局保持", options: { bullet: true } },
  ], { x: 9.4, y: 2.9, w: 3.1, h: 2.2, fontFace: F, fontSize: 13, color: C.ink, lineSpacingMultiple: 1.3, margin: 0 });

  concl(s, "现象是 BASIC + ICC + CG 方法内在,与具体实现无关 —— 后续可在任一实现上做。");
  footer(s, 5);
}

// =========================================================================
// S6 — experiment 2: locate root cause (factor A)
// =========================================================================
{
  const s = P.addSlide(); header(s, "实验 2:定位病根 —— 子域解是否精确", "Sys3 / Laplace");
  s.addText("唯一变量 = 子域求解器精度(BASIC、矩阵、分区全部不变)。nx=48,4 ranks。", {
    x: M, y: 1.5, w: 11.9, h: 0.5, fontFace: F, fontSize: 14.5, color: C.ink, margin: 0 });

  s.addTable([
    [ hc("子域求解器"), hc("O = 0"), hc("O = 1"), hc("O = 2"), hc("趋势") ],
    [ bc("精确 Cholesky"), bc("52"), bc("34"), bc("30", C.green), tc("↓ 下降", "down") ],
    [ bc("ICC(0) 不精确"), bc("132"), bc("148"), bc("179", C.red), tc("↑ 上升", "up") ],
  ], { x: M, y: 2.25, w: 8.3, colW: [3.0, 1.3, 1.3, 1.3, 1.4], rowH: 0.62, fontFace: F, fontSize: 15,
       border: { pt: 0.5, color: C.line }, valign: "middle", align: "center" });

  // ICC-level axis note card
  s.addShape(P.shapes.RECTANGLE, { x: 9.2, y: 2.25, w: 3.43, h: 1.86, fill: { color: "EEF1F7" } });
  s.addText([
    { text: "再加一轴:ICC 填充 L\n", options: { bold: true, color: C.ink } },
    { text: "同 overlap 下 L=0→1→2 迭代单调降 —— 双重锁定「不精确」是病根。", options: { color: C.ink } },
  ], { x: 9.4, y: 2.4, w: 3.05, h: 1.6, fontFace: F, fontSize: 12.5, align: "left", valign: "middle", margin: 0 });

  concl(s, "子域精确求解 ⇒ overlap↑ 迭代↓(恢复经典理论);不精确 ⇒ 反向。病根之一 = 因子 A。");
  footer(s, 6);
}

// =========================================================================
// S7 — experiment 3: fix factor B (sASM)
// =========================================================================
{
  const s = P.addSlide(); header(s, "实验 3:修因子 B —— sASM 对称缩放", "Sys3 / Laplace");
  // formula card
  s.addShape(P.shapes.RECTANGLE, { x: M, y: 1.55, w: 11.93, h: 1.0, fill: { color: C.navy } });
  s.addText("M⁻¹_sASM  =  D⁻¹ᐟ²  ·  M⁻¹_BASIC  ·  D⁻¹ᐟ² ,        D = diag(mₖ)", {
    x: M, y: 1.55, w: 11.93, h: 1.0, fontFace: MONO, fontSize: 19, bold: true, color: "FFFFFF",
    align: "center", valign: "middle", margin: 0 });

  s.addTable([
    [ hc("方案"), hc("O = 0"), hc("O = 1"), hc("O = 2"), hc("趋势") ],
    [ bc("scheme 0  BASIC"), bc("132"), bc("148"), bc("179", C.red), tc("↑ 上升", "up") ],
    [ bc("scheme 3  sASM"), bc("132"), bc("104"), bc("103", C.green), tc("↓ 下降", "down") ],
  ], { x: M, y: 2.95, w: 8.3, colW: [3.0, 1.3, 1.3, 1.3, 1.4], rowH: 0.62, fontFace: F, fontSize: 15,
       border: { pt: 0.5, color: C.line }, valign: "middle", align: "center" });

  s.addShape(P.shapes.RECTANGLE, { x: 9.2, y: 2.95, w: 3.43, h: 1.86, fill: { color: C.greenbg } });
  s.addText([
    { text: "对称 → 保 CG\n", options: { bold: true, color: C.green } },
    { text: "约 10 行 PCSHELL、单层、无额外内存/全局通信。O=0 时 D=I,sASM≡BASIC(自检)。", options: { color: C.ink } },
  ], { x: 9.4, y: 3.1, w: 3.05, h: 1.6, fontFace: F, fontSize: 12.5, align: "left", valign: "middle", margin: 0 });

  concl(s, "用 D⁻¹ᐟ² 精确抵消重叠区重复计数(因子 B),迭代趋势翻回下降,且仍是对称 CG。");
  footer(s, 7);
}

// =========================================================================
// S8 — experiment 4: fix both (scheme 4)
// =========================================================================
{
  const s = P.addSlide(); header(s, "实验 4:sASM + Chebyshev 块解(修两因子)", "Sys3 / Laplace");
  s.addText("子域用 2 步 Chebyshev(以 ICC 为光滑子)廉价逼近精确块逆:零额外内存、零额外全局通信、仍对称保 CG。", {
    x: M, y: 1.5, w: 11.9, h: 0.5, fontFace: F, fontSize: 13.5, color: C.ink, margin: 0 });

  const hr = ["(O, L)", "scheme 0\nBASIC", "scheme 3\nsASM", "scheme 4\nsASM+Cheby2"];
  const rows = [
    ["0, 0", "132", "132", "85"],
    ["1, 0", "148", "104", "68"],
    ["2, 0", "179", "103", "65"],
    ["2, 2", "90",  "67",  "42"],
  ];
  const tbl = [ hr.map(t => hc(t)) ];
  rows.forEach((r, i) => {
    tbl.push([
      bc(r[0]),
      bc(r[1], i >= 1 ? C.red : C.ink),
      bc(r[2], C.ink),
      bc(r[3], C.green, true),
    ]);
  });
  s.addTable(tbl, { x: M, y: 2.2, w: 7.8, colW: [1.5, 2.1, 2.1, 2.1], rowH: 0.6, fontFace: F, fontSize: 14,
    border: { pt: 0.5, color: C.line }, valign: "middle", align: "center" });

  bigStat(s, 9.0, 2.45, "179 → 65", "scheme 4 在 O=2,L=0\n≈ baseline 的 1/3", C.green);
  s.addText("双实现 9/9 逐位一致 · 解正确(误差 5.4e-5)", { x: 9.0, y: 5.1, w: 3.6, h: 0.6,
    fontFace: F, fontSize: 12, color: C.muted, align: "center", margin: 0 });

  concl(s, "A、B 两个因子被同时廉价压低,scheme 4 在每个 (O,L) 都最低,且 overlap↑ 单调下降。");
  footer(s, 8);
}

// =========================================================================
// S9 — universality
// =========================================================================
{
  const s = P.addSlide(); header(s, "普遍性:换两个问题,同现象、同修复");
  // pure Neumann card
  uniCard(s, M, 1.6, "pure Neumann(奇异椭圆)",
    [["BASIC", "64→68→87", "up"], ["sASM", "64→52→51", "down"]],
    "overlap↑ 同样发病;sASM 同样修复。");
  // reaction-diffusion card
  uniCard(s, 6.95, 1.6, "reaction-diffusion(质量主导)",
    [["ICC(0)  心脏 dt", "6→8→9", "up"], ["ICC(2)", "6→6→7", "flat"]],
    "低 ICC 也发病;提高 ICC / sASM 即压平。");

  s.addShape(P.shapes.RECTANGLE, { x: M, y: 4.85, w: 11.93, h: 1.35, fill: { color: C.navy } });
  s.addText([
    { text: "结论   ", options: { bold: true, color: C.ice, fontSize: 16 } },
    { text: "「overlap↑ 迭代↑」不是某个问题独有 —— 只要局部解够不精确(低 ICC),三个问题都发病;差别只是幅度(质量主导→幅度小,椭圆→幅度大)。同一套 sASM / scheme 4 都能修。", options: { color: "FFFFFF", fontSize: 14 } },
  ], { x: M + 0.35, y: 4.85, w: 11.2, h: 1.35, fontFace: F, align: "left", valign: "middle", margin: 0 });
  footer(s, 9);
}

// =========================================================================
// S10 — also reduces time (synchronization)
// =========================================================================
{
  const s = P.addSlide(); header(s, "修复不止降迭代,也降「时间」", "Sys3 / Laplace");
  s.addText("用 -log_view 的全局同步计数(确定性、机器无关;wall-time 在负载机器上不可信)。", {
    x: M, y: 1.5, w: 11.9, h: 0.5, fontFace: F, fontSize: 14, color: C.ink, margin: 0 });

  s.addTable([
    [ hc("方案 (O=2, L=0)"), hc("迭代"), hc("全局同步 (MPI_Allreduce)") ],
    [ bc("scheme 0  baseline"), bc("179", C.red), bc("668", C.red) ],
    [ bc("scheme 4  sASM+Cheby2"), bc("65", C.green), bc("329", C.green) ],
  ], { x: M, y: 2.7, w: 8.0, colW: [3.4, 1.5, 3.1], rowH: 0.82, fontFace: F, fontSize: 15,
       border: { pt: 0.5, color: C.line }, valign: "middle", align: "center" });

  s.addShape(P.shapes.RECTANGLE, { x: 9.0, y: 2.7, w: 3.63, h: 2.46, fill: { color: C.greenbg } });
  s.addText([
    { text: "VecTDot = 2·iter + 2\n", options: { fontFace: MONO, bold: true, color: C.green } },
    { text: "严格成立 ⇒ 内层 Chebyshev(在 SELF 块上)零全局同步;外层迭代降 = 全局同步降。", options: { fontFace: F, color: C.ink } },
  ], { x: 9.2, y: 2.85, w: 3.25, h: 2.16, fontSize: 13, align: "left", valign: "middle", margin: 0 });

  concl(s, "全局 MPI_Allreduce 从 668 降到 329(≈2×)—— 这正是大规模并行(数千 rank)的真瓶颈。");
  footer(s, 10);
}

// =========================================================================
// S11 — literature & theory positioning
// =========================================================================
{
  const s = P.addSlide(); header(s, "文献与理论定位:已证 / 待证");
  // left: literature
  s.addText("现象在文献中", { x: M, y: 1.55, w: 5.6, h: 0.4, fontFace: F, fontSize: 15, bold: true, color: C.teal, margin: 0 });
  s.addText([
    { text: "Cai–Sarkis 1999", options: { bold: true } },
    { text:  " — RAS,指出 BASIC 重叠区重复计数\n", options: {} },
    { text: "Efstathiou–Gander 2003", options: { bold: true } },
    { text: " — 证 AS 收敛依赖着色数\n", options: {} },
    { text: "Ernst–Flemisch–Wohlmuth 2009", options: { bold: true, color: C.red } },
    { text: " — 逐字记录「overlap 再增大,不精确法的迭代数比精确法更差」(异设定,诚实标注)", options: {} },
  ], { x: M, y: 2.05, w: 5.7, h: 2.4, fontFace: F, fontSize: 12.5, color: C.ink, lineSpacingMultiple: 1.15, margin: 0 });

  // right: proven/open table
  s.addText("理论状态", { x: 6.95, y: 1.55, w: 5.6, h: 0.4, fontFace: F, fontSize: 15, bold: true, color: C.teal, margin: 0 });
  s.addTable([
    [ hc("命题"), hc("状态") ],
    [ bc2("λmax ≤ Nc(因子 B)"), bc("已严格证明", C.green) ],
    [ bc2("ω 谱等价(因子 A)"), bc("已严格证明", C.green) ],
    [ bc2("精确解 overlap↑→↓"), bc("已严格证明", C.green) ],
    [ bc2("缩放令 λmax≤1(sASM)"), bc("已严格证明", C.green) ],
    [ bc2("A×B ⇒ 迭代严格上升"), bc("机制+实证,无闭式", C.red) ],
  ], { x: 6.95, y: 2.05, w: 5.68, colW: [3.5, 2.18], rowH: 0.5, fontFace: F, fontSize: 12.5,
       border: { pt: 0.5, color: C.line }, valign: "middle", align: "left" });

  s.addShape(P.shapes.RECTANGLE, { x: M, y: 5.55, w: 11.93, h: 0.85, fill: { color: "EEF1F7" } });
  s.addText([
    { text: "开放问题:", options: { bold: true, color: C.red } },
    { text: "  给出固定填充 ICC 在增大子块上的谱等价常数 ω(δ) 随 overlap 增长的闭式下界 —— 把「机制」补成定理。", options: { color: C.ink } },
  ], { x: M + 0.3, y: 5.55, w: 11.3, h: 0.85, fontFace: F, fontSize: 13, align: "left", valign: "middle", margin: 0 });
  footer(s, 11);
}

// =========================================================================
// S12 — conclusion (dark)
// =========================================================================
{
  const s = P.addSlide(); darkBg(s);
  s.addText("结论", { x: 1.0, y: 0.7, w: 11, h: 0.8, fontFace: F, fontSize: 32, bold: true, color: "FFFFFF", margin: 0 });
  cBlock(s, 1.0, 1.85, "①", "不是 bug", "两套独立实现 27/27 逐位一致,残差历史逐位相同。");
  cBlock(s, 1.0, 3.0,  "②", "病根 = A × B", "子域解不精确(A)× 重叠重复计数(B)的乘积;拔掉任一即翻正。");
  cBlock(s, 1.0, 4.15, "③", "修复 = sASM (+ Chebyshev)", "D⁻¹ᐟ² 缩放消 B,Chebyshev 块解廉价压 A;保 CG、保单层。");

  s.addShape(P.shapes.RECTANGLE, { x: 7.4, y: 1.85, w: 5.2, h: 3.45, fill: { color: C.navy2 } });
  s.addText("普遍性", { x: 7.65, y: 2.05, w: 4.7, h: 0.4, fontFace: F, fontSize: 15, bold: true, color: C.ice, margin: 0 });
  s.addText("三个问题(Laplace / pure-Neumann / reaction-diffusion)在低 ICC 都发病,同一套方法都能修。", {
    x: 7.65, y: 2.5, w: 4.7, h: 1.1, fontFace: F, fontSize: 13.5, color: "FFFFFF", margin: 0 });
  s.addText("开放问题", { x: 7.65, y: 3.75, w: 4.7, h: 0.4, fontFace: F, fontSize: 15, bold: true, color: C.ice, margin: 0 });
  s.addText("ICC 在增大子块上 ω(δ) 的闭式下界 —— 把机制补成定理。", {
    x: 7.65, y: 4.2, w: 4.7, h: 0.9, fontFace: F, fontSize: 13.5, color: "FFFFFF", margin: 0 });

  s.addText("代码 · 数据 · 完整报告:  github.com/tianhaomahpc-ops/cg-asm-icc-research", {
    x: 1.0, y: 6.45, w: 11.5, h: 0.4, fontFace: MONO, fontSize: 13, color: "6FB3CE", margin: 0 });
}

// ---------- small component helpers (defined after use is fine: hoisted) ----------
function hCell()  { return { fill: { color: C.navy }, color: "FFFFFF", bold: true, fontSize: 13 }; }
function bCell(col) { return { color: col || C.ink, fontSize: 14 }; }
function hc(t)  { return { text: t, options: { fill: { color: C.navy }, color: "FFFFFF", bold: true } }; }
function bc(t, col, bold) { return { text: t, options: { color: col || C.ink, bold: !!bold } }; }
function bc2(t) { return { text: t, options: { color: C.ink, align: "left" } }; }
function tc(t, kind) {
  const col = kind === "up" ? C.red : C.green;
  const bg  = kind === "up" ? C.redbg : C.greenbg;
  return { text: t, options: { color: col, bold: true, fill: { color: bg } } };
}
function qBlock(s, x, y, num, head, body) {
  s.addShape(P.shapes.OVAL, { x, y, w: 0.7, h: 0.7, fill: { color: C.teal } });
  s.addText(num, { x, y, w: 0.7, h: 0.7, fontFace: F, fontSize: 22, bold: true, color: "FFFFFF", align: "center", valign: "middle", margin: 0 });
  s.addText(head, { x: x + 0.9, y: y - 0.05, w: 4.6, h: 0.45, fontFace: F, fontSize: 17, bold: true, color: C.ink, margin: 0 });
  s.addText(body, { x: x + 0.9, y: y + 0.42, w: 4.6, h: 0.8, fontFace: F, fontSize: 13, color: C.muted, margin: 0 });
}
function chip(s, x, y, head, body) {
  s.addShape(P.shapes.RECTANGLE, { x, y, w: 4.55, h: 0.62, fill: { color: C.card }, shadow: shadow() });
  s.addShape(P.shapes.RECTANGLE, { x, y, w: 0.08, h: 0.62, fill: { color: C.teal } });
  s.addText([
    { text: head + "   ", options: { bold: true, color: C.ink, fontSize: 13.5 } },
    { text: body, options: { color: C.muted, fontSize: 11.5 } },
  ], { x: x + 0.25, y, w: 4.2, h: 0.62, fontFace: F, align: "left", valign: "middle", margin: 0 });
}
function factorCard(s, x, y, accent, sym, head, body) {
  const w = 3.15;
  s.addShape(P.shapes.RECTANGLE, { x, y, w, h: 1.9, fill: { color: C.card }, shadow: shadow() });
  s.addShape(P.shapes.RECTANGLE, { x, y, w, h: 0.1, fill: { color: accent } });
  s.addText(sym, { x, y: y + 0.18, w, h: 0.55, fontFace: MONO, fontSize: 19, bold: true, color: accent, align: "center", margin: 0 });
  s.addText(head, { x: x + 0.12, y: y + 0.78, w: w - 0.24, h: 0.4, fontFace: F, fontSize: 12.5, bold: true, color: C.ink, align: "center", margin: 0 });
  s.addText(body, { x: x + 0.12, y: y + 1.18, w: w - 0.24, h: 0.62, fontFace: F, fontSize: 10.5, color: C.muted, align: "center", margin: 0 });
}
function bigStat(s, x, y, stat, label, col) {
  s.addShape(P.shapes.RECTANGLE, { x, y, w: 3.7, h: 3.0, fill: { color: C.card }, shadow: shadow() });
  s.addText(stat, { x: x + 0.1, y: y + 0.55, w: 3.5, h: 1.1, fontFace: MONO, fontSize: 40, bold: true, color: col, align: "center", margin: 0 });
  s.addText(label, { x: x + 0.25, y: y + 1.75, w: 3.2, h: 1.0, fontFace: F, fontSize: 13, color: C.ink, align: "center", valign: "top", margin: 0 });
}
function concl(s, txt) {
  s.addShape(P.shapes.RECTANGLE, { x: M, y: 5.75, w: 11.93, h: 0.95, fill: { color: C.greenbg } });
  s.addShape(P.shapes.RECTANGLE, { x: M, y: 5.75, w: 0.1, h: 0.95, fill: { color: C.green } });
  s.addText([
    { text: "结论   ", options: { bold: true, color: C.green } },
    { text: txt, options: { color: C.ink } },
  ], { x: M + 0.35, y: 5.75, w: 11.3, h: 0.95, fontFace: F, fontSize: 14, align: "left", valign: "middle", margin: 0 });
}
function uniCard(s, x, y, title, rows, note) {
  s.addShape(P.shapes.RECTANGLE, { x, y, w: 5.68, h: 3.05, fill: { color: C.card }, shadow: shadow() });
  s.addShape(P.shapes.RECTANGLE, { x, y, w: 5.68, h: 0.1, fill: { color: C.teal } });
  s.addText(title, { x: x + 0.3, y: y + 0.25, w: 5.1, h: 0.45, fontFace: F, fontSize: 15, bold: true, color: C.ink, margin: 0 });
  rows.forEach((r, i) => {
    const yy = y + 0.85 + i * 0.62;
    s.addText(r[0], { x: x + 0.3, y: yy, w: 2.7, h: 0.5, fontFace: F, fontSize: 13.5, color: C.ink, valign: "middle", margin: 0 });
    s.addText(r[1], { x: x + 2.9, y: yy, w: 1.55, h: 0.5, fontFace: MONO, fontSize: 14, bold: true,
      color: r[2] === "up" ? C.red : r[2] === "down" ? C.green : C.muted, valign: "middle", align: "right", margin: 0 });
    s.addText(r[2] === "up" ? "↑" : r[2] === "down" ? "↓" : "~", { x: x + 4.55, y: yy, w: 0.7, h: 0.5,
      fontFace: F, fontSize: 16, bold: true, color: r[2] === "up" ? C.red : r[2] === "down" ? C.green : C.muted, valign: "middle", align: "center", margin: 0 });
  });
  s.addText(note, { x: x + 0.3, y: y + 2.45, w: 5.1, h: 0.5, fontFace: F, fontSize: 11.5, color: C.muted, italic: true, margin: 0 });
}
function cBlock(s, x, y, num, head, body) {
  s.addShape(P.shapes.OVAL, { x, y, w: 0.8, h: 0.8, fill: { color: C.teal } });
  s.addText(num, { x, y, w: 0.8, h: 0.8, fontFace: F, fontSize: 24, bold: true, color: "FFFFFF", align: "center", valign: "middle", margin: 0 });
  s.addText(head, { x: x + 1.0, y: y - 0.02, w: 5.4, h: 0.5, fontFace: F, fontSize: 19, bold: true, color: "FFFFFF", margin: 0 });
  s.addText(body, { x: x + 1.0, y: y + 0.48, w: 6.2, h: 0.7, fontFace: F, fontSize: 13, color: C.ice, margin: 0 });
}

P.writeFile({ fileName: "ASM_overlap_anomaly_zh.pptx" }).then(f => console.log("WROTE", f));
