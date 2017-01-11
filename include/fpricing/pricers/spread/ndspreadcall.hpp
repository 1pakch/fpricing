// ndspreadcall.hpp - spread call pricer for normally distributed log-returns

#ifndef NDSPREADCALL_HPP
#define NDSPREADCALL_HPP

#include <fpricing/math/types.hpp>
#include <fpricing/math/normal.hpp>
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

  CeSpreadCallPayoff(
      const math::distr::Normal<2> logret,
      double strike
  ):
    mu1(logret.mean(0)),
    mu2(logret.mean(1)),
    v1(std::sqrt(logret.cov(0,0))),
    v2(std::sqrt(logret.cov(1,1))),
    rho(logret.cov(0,1)/v1/v2),
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
    double y = std::exp(a+b2/2)*math::stdnormcdf(d1) - c*math::stdnormcdf(d2);
    y *= math::stdnormpdf(z);
    return y;
  }
};

} // namespace detail


/// Expectation of a spread call payoff assuming returns are normal.
/// Note that this function does not account for discounting.
double ndspreadcall(
  math::distr::Normal<2> const& logret_distr,
  double strike
){
  auto integrand = detail::CeSpreadCallPayoff(logret_distr, strike);
  return math::gauss_legendre<32>(integrand, -5, 5);
}


} // namespace fpricing

#endif
