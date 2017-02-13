// normal.hpp - normal distribution

#ifndef NORMAL_HPP
#define NORMAL_HPP

#include <fpricing/math/types.hpp>


namespace fpricing {
namespace math {

/// Cdf of a standard normal random variable.
double stdnormcdf(double x)
{
  return std::erfc(-M_SQRT1_2 * x) / 2;
}

/// Pdf of a standard normal random variable.
double stdnormpdf(double x)
{
  return 1/std::sqrt(2*M_PI) * std::exp(-x*x/2);
}

namespace distr {

/// A multivariate normal distibution
template<int N>
class Normal
{
 public:

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  vector<N> mean;   ///< The mean.
  covmatrix<N> cov; ///< The covariance matrix.

  constexpr double ndim() const { return N; };

  template<typename Mean, typename Cov>
  Normal(Mean mean, Cov cov):
    mean(std::forward<Mean>(mean)),
    cov(std::forward<Cov>(cov))
  {}

  // The distribution resulting from premultiplying by a matrix A.
  template<int K>
  Normal<K> premultiply(matrix<K, N> A) const
  {
    return Normal<K>(A*mean, A*cov*A.transpose());
  }

  // The distribution resulting from adding a constant vector.
  decltype(auto) add(const vector<N>& mu) const
  {
    return Normal(mean + mu, cov);
  }

  // Returns the log Laplace transform.
  komplex log_laplace_transform(cxvector<N> z) const
  {
    return (
      (mean.transpose() * z)(0) +
      0.5 * (z.transpose() * cov * z)(0)
    );
  }

  // Returns the Laplace transform.
  komplex laplace_transform(cxvector<N> z) const
  {
    return std::exp(log_laplace_transform(z));
  }

};


} // namespace distr

} // namespace math

} // namespace fpricing

#endif /* NORMAL_HPP */
