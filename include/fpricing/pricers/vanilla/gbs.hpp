// gbs.hpp - European vanilla

#ifndef GBS_HPP
#define GBS_HPP 

#include <cmath>

#include <fpricing/math/normal.hpp>


namespace fpricing {

/// The expected payoff of a vanilla option when returns are Normal.
double gbs(math::distr::Normal<1> X, double strike)
{
  auto mu = X.mean(0);
  auto sigma2 = X.cov(0, 0);
  auto sigma = std::sqrt(sigma2);
  auto logstrike = std::log(strike);
  auto d1 = (mu + sigma2 - logstrike)/sigma;
  auto d2 = d1 - sigma;
  return (
    std::exp(mu + sigma2/2) * math::stdnormcdf(d1)
    - strike * math::stdnormcdf(d2)
  );
}

} // namespace fpricing

#endif /* GBS_HPP */
