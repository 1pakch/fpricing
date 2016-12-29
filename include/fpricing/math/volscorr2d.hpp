// volscorr2d.hpp - 2x2 covariance matrix as volatilities and the correlation

#ifndef VOLS_CORR_2D_HPP
#define VOLS_CORR_2D_HPP 

#include <eigen3/Eigen/Dense>

#include <fpricing/math/normdist2d.hpp>


namespace fpricing {
namespace math {

/// Two-by-two covariance matrix as volatilities and correlation.
struct VolsCorr2d
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW 

  /// The volatilities.
  const Eigen::Vector2d vols;

  /// The correlation.
  const double corr;

  VolsCorr2d(double vol1, double vol2, double corr):
    vols({vol1, vol2}),
    corr(corr)
  {}

  VolsCorr2d(const Eigen::Matrix2d& cov):
    vols({std::sqrt(cov(0,0)), std::sqrt(cov(1,1))}),
    corr(cov(0,1)/vols(0)/vols(1))
  {}

  /// Convert to a covariance matrix
  Eigen::Matrix2d ToMatrix() const 
  {
    return Eigen::Matrix2d({
      {vols(0)*vols(0), vols(0)*vols(1)*corr},
      {vols(0)*vols(1)*corr, vols(1)*vols(1)}
    });
  }
};

} // namespace math
} // namespace fpricing


#endif /* VOL_CORR_DECOMPOSITION_HPP */
