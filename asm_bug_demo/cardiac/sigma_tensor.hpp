// sigma_tensor.hpp -- anisotropic conductivity tensor as an MFEM
// MatrixCoefficient, switched by element domain attribute.
//
//   heart (attr HEART_ATTR): sigma = R diag(sL, sT, sT) R^T
//                            fibers along x => R = I => diag(sL,sT,sT).
//   torso (attr TORSO_ATTR): sigma = so * I  (isotropic passive conductor).
//
// Passed straight into DiffusionIntegrator(sigma), so anisotropy is a
// one-line integrator swap.  P1 + element-piecewise-constant tensor =>
// quadrature is exact.  Tensor is SPD as long as sL,sT,so > 0 (required or
// ICC on the subdomain blocks breaks).
//
// To rotate fibers later: build R per element from a fiber-direction field
// and return R*diag(sL,sT,sT)*R^T; keep it SPD.
#ifndef SIGMA_TENSOR_HPP
#define SIGMA_TENSOR_HPP

#include "mfem.hpp"
using namespace mfem;

class SigmaTensor : public MatrixCoefficient
{
    const int    heart_attr, torso_attr;
    const double sL, sT;   // heart longitudinal / transverse (fiber || x)
    const double so;       // torso isotropic
public:
    SigmaTensor(int heart_attr_, int torso_attr_,
                double sL_, double sT_, double so_)
        : MatrixCoefficient(3),
          heart_attr(heart_attr_), torso_attr(torso_attr_),
          sL(sL_), sT(sT_), so(so_) {}

    void Eval(DenseMatrix &K, ElementTransformation &T,
              const IntegrationPoint &ip) override
    {
        K.SetSize(3);
        K = 0.0;
        if (T.Attribute == heart_attr)
        {
            K(0,0) = sL;  K(1,1) = sT;  K(2,2) = sT;   // fibers along x
        }
        else // torso_attr (or anything else): isotropic
        {
            K(0,0) = so;  K(1,1) = so;  K(2,2) = so;
        }
    }
};

#endif // SIGMA_TENSOR_HPP
