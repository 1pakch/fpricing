// normdist2d - bivariate normal distribution.

#ifndef NORMDIST2D_HPP
#define NORMDIST2D_HPP 

#include <eigen3/Eigen/Dense>


namespace fpricing {
namespace math {


/// Bivariate normal distribution.
struct NormDist2d
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW 

    /// The mean.
    const Eigen::Vector2d mean;

    /// The covariance matrix.
    const Eigen::Matrix2d cov;

    // Constructor using perfect forwarding.
    template<typename Mean, typename Cov>
    NormDist2d(Mean mean, Cov cov):
        mean(std::forward<Mean>(mean)),
        cov(std::forward<Cov>(cov))
    {}
};


} // namespace math
} // namespace fpricing

#endif /* NORMDIST2D_HPP */
