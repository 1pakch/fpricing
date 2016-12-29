// bb76.hpp - bivariate Black-76 model.

#ifndef BB76_HPP
#define BB76_HPP 

#include <eigen3/Eigen/Dense>

#include <fpricing/math/normdist2d.hpp>


namespace fpricing {

/// Bivariate extension of Black-76 model.
struct BB76
{
  const double vol1; /// Instantaneous volatility of the first futures
  const double vol2; /// Instantaneous volatility of the second futures
  const double corr; /// Instantaneous correlation

  BB76(double v1, double v2, double corr):
    vol1(v1),
    vol2(v2),
    corr(corr)
  {}

  /// State of the model - current log prices.
  class State: public Eigen::Vector2d
  {
    // Vector2d is a typedef of Matrix<double, 2, 1>
    using Eigen::Vector2d::Vector2d;
   public:
    static State FromPrices(double F1, double F2)
    {
      return State({std::log(F1), std::log(F2)});
    }
    static State FromLogPrices(double f1, double f2)
    {
      return State({f1, f2});
    };
  };
  
  /// The distribution of returns at a given horizon.
  math::NormDist2d DistributionOfReturns(State state, double tau)
  {
      auto cov = tau * math::VolsCorr2d(vol1, vol2, corr).ToMatrix();
      auto mu = -0.5*cov.diagonal() + state;
      return math::NormDist2d(mu, cov);
  }
};

} // namespace fpricing

#endif /* BB76_HPP */
