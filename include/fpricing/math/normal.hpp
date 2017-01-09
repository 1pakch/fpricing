// normal.hpp - multivariate Normal distribution

#ifndef NORMAL_HPP
#define NORMAL_HPP

#include <fpricing/math/types.hpp>


namespace fpricing {
namespace math {
namespace distr {


/// A multivariate normal distibution
template<int N>
class Normal
{
 public:

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  vector<N> mean; ///< The mean.
  covmatrix<N> cov; ///< The covariance matrix.

  template<typename Mean, typename Cov>
  Normal(Mean mean, Cov cov):
    mean(std::forward<Mean>(mean)),
    cov(std::forward<Cov>(cov))
  {}

  // The distribution resulting from premultiplying by a matrix A.
  decltype(auto) premultiply(const Eigen::MatrixXd& A)
  {
    return Normal(A*mean, A*cov*A.transpose());
  }

  // The distribution resulting from adding a constant vector
  decltype(auto) add(const vector<N>& mu)
  {
    return Normal(mean + mu, cov);
  }

  // Returns the log Laplace transform function.
  decltype(auto) get_log_laplace_transform()
  {
    auto mean_ = mean;
    auto cov_ = cov;
    return [mean_, cov_](cxvector<N> z){
      return (
          mean_.transpose() * z +
          0.5 * (z.transpose() * cov_ * z)(0)
      );
    };
  }

};


} // namespace distr

} // namespace math

} // namespace fpricing

#endif /* NORMAL_HPP */
