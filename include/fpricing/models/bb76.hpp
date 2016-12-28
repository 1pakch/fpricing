// bb76.hpp - bivariate Black-76 model.

#ifndef BB76_HPP
#define BB76_HPP 

#include <fpricing/math/normdist2d.hpp>


namespace fpricing {

/// Bivariate extension of Black-76 model.
struct BB76
{
  const double vol1;
  const double vol2;
  const double corr;

  BB76(double v1, double v2, double corr):
    vol1(v1),
    vol2(v2),
    corr(corr)
  {}

  math::NormDist2d DistributionOfReturns(double F1, double F2, double tau)
  {
      Eigen::Vector2d mu;
      Eigen::Matrix2d cov;
      cov = tau * math::VolsCorr2d(vol1, vol2, corr).ToMatrix();
      mu = -0.5*cov.diagonal();
      mu(0) += std::log(F1);
      mu(1) += std::log(F2);
      return math::NormDist2d(mu, cov);
  }
};

} // namespace fpricing

#endif /* BB76_HPP */
