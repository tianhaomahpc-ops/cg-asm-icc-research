// soras_demo.cpp -- proper optimized Schwarz (SORAS-type) with the REAL Neumann
// local matrices, to realize the lambda_min gain the purely-algebraic -robin
// could not (see REPORT 1.5.1).
//
// Key point: PCASM hands you A_i = R_i A R_i^T, the DIRICHLET block (cut
// couplings = outside set to 0).  Adding a positive diagonal there only
// over-pins it -> no Robin.  Here we instead assemble, per subdomain, the
// NEUMANN block K_i (element matrices restricted to the subdomain's elements,
// natural BC on the artificial interface).  For a "floating" interior
// subdomain K_i is SINGULAR (pure-Neumann), which is exactly why a plain
// Neumann-Neumann method fails -- and why the Robin term alpha*M_Gamma
// (boundary mass on the artificial interface) is added: it REGULARIZES the
// floating block AND encodes the optimized transmission condition.
//
//   one-level additive Schwarz preconditioner (CG-valid, symmetric):
//     M^{-1} = sum_i  R_i^T D_i ( K_i + alpha M_Gamma,i )^{-1} D_i R_i
//   D_i = 1/multiplicity (partition of unity).  Non-overlapping (minimal FEM
//   overlap = the shared interface layer).
//
//   -dirichlet : use the Dirichlet principal submatrix R_i A R_i^T instead
//                (alpha ignored) -> classic one-level scaled ASM, for contrast.
//
// Build (in-container):
//   mpic++ -O3 -std=c++17 -I/opt/mfem-4.9 -I. -I/usr/include/hypre \
//     -I<petsc>/include soras_demo.cpp -o soras_demo -L/opt/mfem-4.9 -lmfem \
//     <hypre> <metis> <petsc> ...   (same flags as forward_ecg)
//
//   mpirun -n 1 ./soras_demo -nx 24 -nsub 8 -alpha 4     # SORAS
//   mpirun -n 1 ./soras_demo -nx 24 -nsub 8 -dirichlet   # Dirichlet ASM
#include "mfem.hpp"
#include <petscksp.h>
#include <vector>
#include <iostream>
#include <cmath>
using namespace mfem;
using namespace std;

// MFEM serial SparseMatrix -> PETSc SeqAIJ (finalized CSR copy).
static Mat ToSeqAIJ(SparseMatrix &S)
{
    S.Finalize();
    const int n = S.Height();
    const int *I = S.GetI(), *J = S.GetJ();
    const double *Aval = S.GetData();
    std::vector<PetscInt> d_nnz(n);
    for (int i = 0; i < n; ++i) d_nnz[i] = I[i+1] - I[i];
    Mat M;
    MatCreateSeqAIJ(PETSC_COMM_SELF, n, n, 0, d_nnz.data(), &M);
    MatSetOption(M, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
    std::vector<PetscInt> cols; std::vector<PetscScalar> vals;
    for (int i = 0; i < n; ++i) {
        cols.clear(); vals.clear();
        for (int k = I[i]; k < I[i+1]; ++k) { cols.push_back(J[k]); vals.push_back(Aval[k]); }
        PetscInt r = i, m = (PetscInt)cols.size();
        if (m) MatSetValues(M, 1, &r, m, cols.data(), vals.data(), INSERT_VALUES);
    }
    MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY);
    return M;
}

// One subdomain's local Robin solver + restriction to global dofs.
struct Sub {
    KSP   ksp;             // (K_i + alpha M_Gamma)^{-1} via Cholesky
    Vec   rloc, zloc;      // local work vectors
    std::vector<PetscInt> g2l_glob;  // global dof index for each local dof
    std::vector<double>   d;         // partition-of-unity weight per local dof
    int   n;
};

struct ShellCtx { std::vector<Sub> *subs; Vec tmp; };

static PetscErrorCode ShellApply(PC pc, Vec r, Vec z)
{
    ShellCtx *c; PCShellGetContext(pc, (void**)&c);
    VecZeroEntries(z);
    const PetscScalar *ra; VecGetArrayRead(r, &ra);
    PetscScalar *za; VecGetArray(z, &za);
    for (auto &s : *c->subs) {
        // R_i r, then D_i
        PetscScalar *rl; VecGetArray(s.rloc, &rl);
        for (int l = 0; l < s.n; ++l) rl[l] = ra[s.g2l_glob[l]] * s.d[l];
        VecRestoreArray(s.rloc, &rl);
        KSPSolve(s.ksp, s.rloc, s.zloc);          // local Robin solve
        const PetscScalar *zl; VecGetArrayRead(s.zloc, &zl);
        for (int l = 0; l < s.n; ++l) za[s.g2l_glob[l]] += zl[l] * s.d[l];  // D_i then R_i^T
        VecRestoreArrayRead(s.zloc, &zl);
    }
    VecRestoreArray(z, &za);
    VecRestoreArrayRead(r, &ra);
    return 0;
}

int main(int argc, char *argv[])
{
    int nx = 24, nsub = 8; double alpha = 4.0; bool dirichlet = false;
    bool pin = false;
    for (int i = 1; i < argc; ++i) {
        string a = argv[i];
        if (a == "-nx" && i+1<argc) nx = atoi(argv[++i]);
        else if (a == "-nsub" && i+1<argc) nsub = atoi(argv[++i]);
        else if (a == "-alpha" && i+1<argc) alpha = atof(argv[++i]);
        else if (a == "-dirichlet") dirichlet = true;
        else if (a == "-pin") pin = true;   // pin one dof (non-singular) instead of nullspace
    }
    PetscInitialize(&argc, &argv, NULL, NULL);

    // ---- serial cube, H1 order 1, pure-Neumann diffusion (Sys2-like) -------
    Mesh mesh = Mesh::MakeCartesian3D(nx, nx, nx, Element::TETRAHEDRON);
    H1_FECollection fec(1, 3);
    FiniteElementSpace fes(&mesh, &fec);
    const int N = fes.GetTrueVSize();

    ConstantCoefficient one(1.0);
    BilinearForm a(&fes);
    a.AddDomainIntegrator(new DiffusionIntegrator(one));
    a.Assemble();
    Array<int> ess;                     // pure Neumann: no essential dofs
    if (pin) { ess.Append(0); }         // optionally pin dof 0
    SparseMatrix Kg; a.FormSystemMatrix(ess, Kg);
    PetscPrintf(PETSC_COMM_SELF, "  [global] Kg %dx%d MaxNorm=%.3e nnz=%d\n",
        Kg.Height(), Kg.Width(), (double)Kg.MaxNorm(), Kg.NumNonZeroElems());
    Mat A = ToSeqAIJ(Kg);
    { PetscReal an; MatNorm(A,NORM_FROBENIUS,&an);
      PetscPrintf(PETSC_COMM_SELF,"  [global] A(petsc) ||.||_F=%.3e\n",(double)an); }
    if (!pin) {  // attach constant nullspace for the outer CG
        MatNullSpace nsp; MatNullSpaceCreate(PETSC_COMM_SELF, PETSC_TRUE, 0, NULL, &nsp);
        MatSetNullSpace(A, nsp); MatNullSpaceDestroy(&nsp);
    }

    // ---- partition elements into nsub, set attributes ----------------------
    int *part = mesh.GeneratePartitioning(nsub, 1);
    for (int e = 0; e < mesh.GetNE(); ++e) mesh.SetAttribute(e, part[e]+1);
    mesh.SetAttributes();

    // ---- multiplicity of each global dof (how many subdomains contain it) --
    std::vector<int> mult(N, 0);
    const int parent_bdr_max = mesh.bdr_attributes.Size() ? mesh.bdr_attributes.Max() : 1;
    const int art_attr = parent_bdr_max + 1;   // SubMesh tags the artificial cut here

    // ---- PASS 1: submeshes, dof maps, multiplicity, local MFEM matrices ----
    std::vector<Sub> subs(nsub);
    std::vector<SubMesh*> sms(nsub, nullptr);
    std::vector<FiniteElementSpace*> sfeses(nsub, nullptr);
    std::vector<SparseMatrix*> Aloc_mfem(nsub, nullptr);   // Neumann + alpha*M_Gamma
    ConstantCoefficient acoef(alpha);
    for (int p = 0; p < nsub; ++p) {
        Array<int> dom(1); dom[0] = p+1;
        sms[p] = new SubMesh(SubMesh::CreateFromDomain(mesh, dom));
        SubMesh &sm = *sms[p];
        FiniteElementSpace *sfes = new FiniteElementSpace(&sm, &fec); sfeses[p] = sfes;
        const int nl = sfes->GetTrueVSize();
        const Array<int> &pv = sm.GetParentVertexIDMap();
        subs[p].g2l_glob.resize(nl); subs[p].n = nl; subs[p].d.resize(nl);
        for (int l = 0; l < nl; ++l) { subs[p].g2l_glob[l] = pv[l]; mult[pv[l]]++; }
        if (!dirichlet) {
            // Neumann stiffness (natural BC at the artificial interface)
            BilinearForm Kf(sfes);
            Kf.AddDomainIntegrator(new DiffusionIntegrator(one));
            Kf.Assemble(); Kf.Finalize();
            // proper Robin term: alpha * boundary mass on the artificial interface
            // (submesh boundary attribute art_attr), assembled as a real M_Gamma.
            Array<int> bmark(sm.bdr_attributes.Size() ? sm.bdr_attributes.Max() : 1);
            bmark = 0;
            if (bmark.Size() >= art_attr) bmark[art_attr-1] = 1;
            BilinearForm Mf(sfes);
            Mf.AddBoundaryIntegrator(new BoundaryMassIntegrator(acoef), bmark);
            Mf.Assemble(); Mf.Finalize();
            SparseMatrix *L = new SparseMatrix(Kf.SpMat());   // copy Neumann block
            L->Add(1.0, Mf.SpMat());                          // += alpha*M_Gamma
            Aloc_mfem[p] = L;
            if (p == 0) PetscPrintf(PETSC_COMM_SELF,
                "  [sub0] NE=%d nl=%d  ||K||=%.3e  ||alpha*M_Gamma||=%.3e  iface_dofs~%d\n",
                sm.GetNE(), nl, (double)Kf.SpMat().MaxNorm(),
                (double)Mf.SpMat().MaxNorm(), Mf.SpMat().NumNonZeroElems());
        }
    }

    // ---- PASS 2: build PETSc local operator + PU weights + local solver ----
    for (int p = 0; p < nsub; ++p) {
        Sub &s = subs[p];
        const int nl = s.n;
        for (int l = 0; l < nl; ++l) s.d[l] = 1.0 / (double)mult[s.g2l_glob[l]];
        Mat Aloc;
        if (dirichlet) {
            // Dirichlet principal submatrix R_p A R_p^T (extract from global A)
            IS is; ISCreateGeneral(PETSC_COMM_SELF, nl, s.g2l_glob.data(),
                                   PETSC_COPY_VALUES, &is);
            Mat *sub; MatCreateSubMatrices(A, 1, &is, &is, MAT_INITIAL_MATRIX, &sub);
            Aloc = sub[0]; PetscFree(sub); ISDestroy(&is);
        } else {
            Aloc = ToSeqAIJ(*Aloc_mfem[p]);     // Neumann + alpha*M_Gamma
        }
        // local solver on the SPD local block: robust iterative CG+ICC
        // (direct PETSc Cholesky on SeqAIJ is fragile; CG+ICC always works on
        // an SPD block and a tight tol makes it effectively exact).
        PetscReal lnorm; MatNorm(Aloc, NORM_FROBENIUS, &lnorm);
        PetscInt lm,ln; MatGetSize(Aloc,&lm,&ln);
        if (p == 0) PetscPrintf(PETSC_COMM_SELF,
            "  [sub0] local block %dx%d  ||.||_F=%.3e\n",(int)lm,(int)ln,(double)lnorm);
        KSP k; KSPCreate(PETSC_COMM_SELF, &k);
        KSPSetType(k, KSPCG); KSPSetOperators(k, Aloc, Aloc);
        KSPSetTolerances(k, 1e-10, 1e-14, PETSC_DEFAULT, 500);
        KSPSetNormType(k, KSP_NORM_UNPRECONDITIONED);
        PC pc; KSPGetPC(k, &pc); PCSetType(pc, PCICC);
        KSPSetErrorIfNotConverged(k, PETSC_FALSE);
        s.ksp = k;
        MatCreateVecs(Aloc, &s.rloc, &s.zloc);
    }
    int max_mult = 0; for (int v=0; v<N; ++v) max_mult = std::max(max_mult, mult[v]);

    // ---- outer CG with the additive-Schwarz shell --------------------------
    ShellCtx ctx; ctx.subs = &subs;
    KSP ksp; KSPCreate(PETSC_COMM_SELF, &ksp);
    KSPSetType(ksp, KSPCG); KSPSetOperators(ksp, A, A);
    KSPSetNormType(ksp, KSP_NORM_PRECONDITIONED);
    KSPSetTolerances(ksp, 1e-6, 1e-50, PETSC_DEFAULT, 2000);
    PC pc; KSPGetPC(ksp, &pc); PCSetType(pc, PCSHELL);
    PCShellSetContext(pc, &ctx); PCShellSetApply(pc, ShellApply);
    KSPSetFromOptions(ksp);

    Vec b, x; MatCreateVecs(A, &b, &x);
    VecSetRandom(b, NULL);
    if (!pin) {   // make RHS compatible (zero mean) for the singular system
        PetscScalar s; VecSum(b, &s); VecShift(b, -s/(double)N);
    }
    // sanity probe: ||M^{-1} b|| should be O(1), not ~0
    { Vec z; VecDuplicate(b, &z); ShellApply(pc, b, z);
      PetscReal zn, bn0; VecNorm(z,NORM_2,&zn); VecNorm(b,NORM_2,&bn0);
      PetscPrintf(PETSC_COMM_SELF, "  [probe] ||b||=%.3e  ||M^-1 b||=%.3e\n",
                  (double)bn0,(double)zn); VecDestroy(&z); }
    VecZeroEntries(x);
    KSPSolve(ksp, b, x);
    PetscInt its; KSPGetIterationNumber(ksp, &its);
    // true residual
    Vec res; VecDuplicate(b, &res); MatMult(A, x, res); VecAYPX(res, -1.0, b);
    PetscReal rn, bn; VecNorm(res, NORM_2, &rn); VecNorm(b, NORM_2, &bn);
    KSPConvergedReason reason; KSPGetConvergedReason(ksp, &reason);

    PetscPrintf(PETSC_COMM_SELF,
        "[SORAS] %s  nx=%d nsub=%d alpha=%g  max_mult=%d  N=%d  iters=%d  "
        "true_res=%.3e  reason=%d %s\n",
        dirichlet ? "DIRICHLET-block" : "NEUMANN+Robin ", nx, nsub, alpha,
        max_mult, N, (int)its, (double)(rn/bn), (int)reason,
        reason>0 ? "(CONVERGED)" : "(DIVERGED)");

    PetscFinalize();
    return 0;
}
