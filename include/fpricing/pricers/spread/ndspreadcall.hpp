// ndspreadcall.hpp - spread call pricer for normally distributed log-returns

#ifndef NDSPREADCALL_HPP
#define NDSPREADCALL_HPP

#include <eigen3/Eigen/Dense>
#include <boost/math/distributions/normal.hpp>

#include <fpricing/math/normdist2d.hpp>
#include <fpricing/math/volscorr2d.hpp>
#include <fpricing/math/gauss_legendre.hpp>


namespace fpricing {
namespace detail {

/// Conditional expectation of a spread call payoff when returns are Normal.
/// The conditioning variable is the standardized return on the second asset.
struct CeSpreadCallPayoff
{
  const double mu1, mu2;
  const double v1, v2;
  const double rho;
  const double strike;
  const double b;
  const double b2;
  boost::math::normal stdnorm;

  CeSpreadCallPayoff(
      const math::NormDist2d dreturns,
      double strike
  ):
    mu1(dreturns.mean(0)),
    mu2(dreturns.mean(1)),
    v1(std::sqrt(dreturns.cov(0,0))),
    v2(std::sqrt(dreturns.cov(1,1))),
    rho(dreturns.cov(0,1)/v1/v2),
    strike(strike),
    b(std::sqrt(1-rho*rho)*v1),
    b2(b*b)
  {}

  double operator() (double z) const 
  {
      double a = mu1 + v1*rho*z;
      double c = strike + std::exp(mu2+v2*z);
      double d1 = (a + b2 - std::log(c))/b;
      double d2 = d1 - b;
      double y = std::exp(a+b2/2)*cdf(stdnorm, d1) - c*cdf(stdnorm, d2);
      y *= pdf(stdnorm, z);
      return y;
  }
};

} // namespace detail


/// Expectation of a spread call payoff assuming returns are normal.
/// Note that this function does not account for discounting.
double ndspreadcall(
  math::NormDist2d const& returns,
  double strike
){
  auto integrand = detail::CeSpreadCallPayoff(returns, strike);
  return math::gauss_legendre<32>(integrand, -5, 5);
}


} // namespace fpricing

#endif
