// interface_transfer.hpp -- couple two INDEPENDENT ParMeshes (heart, torso)
// that share a conforming interface surface, transferring a scalar field on the
// interface ONLY, by coordinate match.  Replaces MFEM ParSubMesh/ParTransferMap
// (whose SubMesh->SubMesh transfer is partition-dependent in this build).
//
// Requirement: the two meshes' interface boundary nodes have IDENTICAL
// coordinates (guaranteed here because heart.msh and torso.msh are written from
// the same Gmsh BooleanFragments mesh -- see heart_torso.py).
//
// Parallel-safe: each interface true-dof is owned by exactly one rank; we
// MPI_Allgatherv the (small) interface values to all ranks and scatter to the
// destination interface true-dofs by a precomputed coordinate match.  The
// result is INDEPENDENT of the partitioning (bit-identical serial vs parallel).
#ifndef INTERFACE_TRANSFER_HPP
#define INTERFACE_TRANSFER_HPP

#include "mfem.hpp"
#include <vector>
#include <map>
#include <tuple>
#include <cmath>
using namespace mfem;

class InterfaceTransfer
{
    MPI_Comm comm;
    Array<int> src_td, dst_td;          // local interface true-dof indices
    std::vector<int> counts, displs;    // Allgatherv layout of the src interface
    int nsrc_glob = 0;
    std::vector<int> dst_match;          // dst_match[k] -> global src index for dst_td[k]
    long long matched = 0, dst_total = 0;

    static void IfaceTDofs(ParFiniteElementSpace &fes, int attr, Array<int> &td)
    {
        const int nb = fes.GetParMesh()->bdr_attributes.Size() ?
                       fes.GetParMesh()->bdr_attributes.Max() : 0;
        Array<int> bdr(nb); bdr = 0;
        if (attr >= 1 && attr <= nb) bdr[attr-1] = 1;
        fes.GetEssentialTrueDofs(bdr, td);
    }
    static void TDofCoords(ParFiniteElementSpace &fes, Vector &X, Vector &Y, Vector &Z)
    {
        ParGridFunction gx(&fes), gy(&fes), gz(&fes);
        FunctionCoefficient fx([](const Vector&p){return p[0];});
        FunctionCoefficient fy([](const Vector&p){return p[1];});
        FunctionCoefficient fz([](const Vector&p){return p[2];});
        gx.ProjectCoefficient(fx); gy.ProjectCoefficient(fy); gz.ProjectCoefficient(fz);
        gx.GetTrueDofs(X); gy.GetTrueDofs(Y); gz.GetTrueDofs(Z);
    }
    static std::tuple<long,long,long> Key(double x,double y,double z)
    {   // quantize to 1e-4 mm; interface coords are bit-identical across files
        return std::make_tuple((long)llround(x*1e4),(long)llround(y*1e4),(long)llround(z*1e4));
    }

public:
    InterfaceTransfer(ParFiniteElementSpace &src, int src_attr,
                      ParFiniteElementSpace &dst, int dst_attr)
    {
        comm = src.GetComm();
        int nranks; MPI_Comm_size(comm, &nranks);
        IfaceTDofs(src, src_attr, src_td);
        IfaceTDofs(dst, dst_attr, dst_td);
        Vector sX,sY,sZ,dX,dY,dZ;
        TDofCoords(src, sX,sY,sZ);  TDofCoords(dst, dX,dY,dZ);

        // gather all source interface coordinates to every rank
        int nloc = src_td.Size();
        counts.assign(nranks,0); displs.assign(nranks,0);
        MPI_Allgather(&nloc,1,MPI_INT,counts.data(),1,MPI_INT,comm);
        for (int r=0;r<nranks;++r){ displs[r]=nsrc_glob; nsrc_glob+=counts[r]; }
        std::vector<double> lx(nloc),ly(nloc),lz(nloc);
        for (int k=0;k<nloc;++k){ int p=src_td[k]; lx[k]=sX(p); ly[k]=sY(p); lz[k]=sZ(p); }
        std::vector<double> gx(nsrc_glob),gy(nsrc_glob),gz(nsrc_glob);
        MPI_Allgatherv(lx.data(),nloc,MPI_DOUBLE,gx.data(),counts.data(),displs.data(),MPI_DOUBLE,comm);
        MPI_Allgatherv(ly.data(),nloc,MPI_DOUBLE,gy.data(),counts.data(),displs.data(),MPI_DOUBLE,comm);
        MPI_Allgatherv(lz.data(),nloc,MPI_DOUBLE,gz.data(),counts.data(),displs.data(),MPI_DOUBLE,comm);

        std::map<std::tuple<long,long,long>,int> idx;
        for (int g=0; g<nsrc_glob; ++g) idx[Key(gx[g],gy[g],gz[g])] = g;

        dst_match.assign(dst_td.Size(), -1);
        long long loc_matched=0;
        for (int k=0;k<dst_td.Size();++k){ int p=dst_td[k];
            auto it = idx.find(Key(dX(p),dY(p),dZ(p)));
            if (it!=idx.end()){ dst_match[k]=it->second; ++loc_matched; }
        }
        long long loc_dst = dst_td.Size();
        MPI_Allreduce(&loc_matched,&matched,1,MPI_LONG_LONG,MPI_SUM,comm);
        MPI_Allreduce(&loc_dst,&dst_total,1,MPI_LONG_LONG,MPI_SUM,comm);
    }

    // src_tv: source true-dof vector; writes the matched interface entries of
    // dst_tv (caller zeroes/uses dst_tv as the Dirichlet lift).
    void Transfer(const Vector &src_tv, Vector &dst_tv) const
    {
        int nloc = src_td.Size();
        std::vector<double> loc(nloc);
        for (int k=0;k<nloc;++k) loc[k] = src_tv(src_td[k]);
        std::vector<double> g(nsrc_glob);
        MPI_Allgatherv(loc.data(),nloc,MPI_DOUBLE,g.data(),counts.data(),displs.data(),MPI_DOUBLE,comm);
        for (int k=0;k<(int)dst_match.size();++k)
            if (dst_match[k]>=0) dst_tv(dst_td[k]) = g[dst_match[k]];
    }

    const Array<int>& DstInterfaceTDofs() const { return dst_td; }
    long long Matched() const { return matched; }
    long long DstTotal() const { return dst_total; }
};

#endif // INTERFACE_TRANSFER_HPP
