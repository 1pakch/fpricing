// bb76.hpp - bivariate Black-76 model.

#ifndef BB76_HPP
#define BB76_HPP

#include <fpricing/math/types.hpp>
#include <fpricing/math/normal.hpp>


namespace fpricing {


/// Bivariate extension of Black-76 model.
struct BB76
{
  const math::covmatrix<2> cov_; ///< Instantaneous covariance matrix.

  BB76(const math::covmatrix<2>& cov):
    cov_(cov)
  {}

  /// State of the model - current log prices.
  class State: public math::vector<2>
  {
    // vector is a template typedef
    using math::vector<2>::vector;
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
  math::distr::Normal<2> DistributionOfReturns(State state, double tau)
  {
      auto cov_tau = tau * cov_;
      auto mu_tau = -0.5 * cov_tau.diagonal() + state;
      return math::distr::Normal<2>(mu_tau, cov_tau);
  }
};

} // namespace fpricing

#endif /* BB76_HPP */
