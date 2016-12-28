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
  Eigen::Vector2d vols;

  /// The correlation.
  double corr;

  VolsCorr2d(double vol1, double vol2, double corr):
    vols(std::vector<double>({vol1, vol2}).data()),
    corr(corr)
  {}

  VolsCorr2d(const Eigen::Matrix2d& cov)
  {
    vols(0) = std::sqrt(cov(0,0));
    vols(1) = std::sqrt(cov(1,1));
    corr = cov(0,1)/vols(0)/vols(1);
  }

  Eigen::Matrix2d ToMatrix() const 
  {
    Eigen::Matrix2d cov;
    cov(0,0) = vols(0)*vols(0);
    cov(1,1) = vols(1)*vols(1);
    cov(0,1) = vols(0)*vols(1)*corr;
    cov(1,0) = cov(0,1);
    return cov;
  }
};

} // namespace math
} // namespace fpricing


#endif /* VOL_CORR_DECOMPOSITION_HPP */
