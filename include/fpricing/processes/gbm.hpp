// gbm.hpp - Multivariate Geometric Brownian Motion.

#ifndef GBM_HPP
#define GBM_HPP 

#include <fpricing/math/types.hpp>
#include <fpricing/math/normal.hpp>


namespace fpricing {


/// Transform Coefficients
template<int N>
struct TransformCoefficients
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  
  const math::komplex const_term {};
  
  const math::cxvector<N> linear_term {};

  math::komplex log_transform(math::vector<N> state) const
  {
    return const_term + state.dot(linear_term);
  }
};


/// A multivariate Geometric Brownian Motion.
template<int N>
class GeometricBrownianMotion
{
 public:

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /// Drift specification
  const math::vector<N> drift {};

  /// Instantaneous covariance matrix
  const math::covmatrix<N> cov {};

  /// GBM with given drift and covariance matrix.
  GeometricBrownianMotion(
      const math::vector<N>& drift,
      const math::covmatrix<N>& cov
  ):
      drift(drift), cov(cov)
  {}

  /// Martingale GBM with given covariance matrix.
  GeometricBrownianMotion(
      const math::covmatrix<N>& cov
  ):
      drift(-0.5*cov.diagonal()), cov(cov)
  {}

  /// Distribution of log returns
  math::distr::Normal<N> log_returns_distr(
      double t,
      double T,
      math::vector<N> state
  ) const
  {
    return math::distr::Normal<N>(
        state + (T-t) * drift,
        (T-t) * cov
    );
  }

  /// Transform coefficients
  decltype(auto) transform_coefficients(
      double t,
      double T,
      math::cxvector<N> z
  ) const
  {
    return TransformCoefficients<N>{
      (T-t) * ((drift * z) + 0.5*(cov * z).dot(z)),
      z
    };
  }
};


//
// Specific Gaussian processes
//

/// Univariate Black-76 returns process.
struct Black76Process:
    public GeometricBrownianMotion<1>
{
  Black76Process(double vol):
    GeometricBrownianMotion<1>(
      math::covmatrix<1>({vol*vol})
    )
  {}
};

/// Two correlated Black-76 returns processes.
struct BivariateBlack76Process:
    public GeometricBrownianMotion<2>
{
  BivariateBlack76Process(double vol1, double vol2, double corr):
    GeometricBrownianMotion<2>(
      math::covmatrix<2>::from_vols_and_corr(vol1, vol2, corr)
    )
  {}
};

} // namespace fpricing


#endif /* GBM_HPP */
