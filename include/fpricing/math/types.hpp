// types.hpp - typedefs for commonly used types and the covmatrix template

#ifndef TYPES_HPP
#define TYPES_HPP 

#include <eigen3/Eigen/Dense>
#include <complex>


namespace fpricing {
namespace math {

// 
// Typedefs for std::complex and Eigen objects
//

using komplex = std::complex<double>;

template<int m, int n>
using matrix = Eigen::Matrix<double, m, n>;

template<int m>
using vector = Eigen::Matrix<double, m, 1>;

template<int m, int n>
using cxmatrix = Eigen::Matrix<komplex, m, n>;

template<int m>
using cxvector = Eigen::Matrix<komplex, m, 1>;

// 
// Specialized matrix class for covariance matrices.
//

namespace detail {

/// Covariance matrix - base class.
template<int N>
class covmatrix_base: public matrix<N, N>
{
 public:
  using matrix<N, N>::matrix;

  /// Construct from volatilities and correlations
  covmatrix_base(vector<N> volatilities, matrix<N, N> correlations):
    matrix<N, N>::matrix(
      volatilities().asDiagonal()
        * correlations
        * volatilities().asDiagonal()
    )
  {}

  /// Volatilities as a vector.
  vector<N> get_volatilities() const {
    return this->diagonal().array().sqrt().matrix();
  }

  /// Corelations matrix using precomputed volatilities.
  matrix<N, N> get_correlations(const vector<N>& vols) const {
    auto invVols = vols.asDiagonal().inverse();
    return invVols * (*this) * invVols;    
  }

  /// Correlations matrix.
  matrix<N, N> get_correlations() const {
    return get_correlations(this->get_volatilities()); 
  }
};

} // namespace detail


/// Covariance matrix - generic version.
template<int N>
class covmatrix: public detail::covmatrix_base<N>
{
 public:
  using detail::covmatrix_base<N>::covmatrix_base;
};


/// Covariance matrix (1x1).
template<>
class covmatrix<1>: public detail::covmatrix_base<1>
{
 public:
  using detail::covmatrix_base<1>::covmatrix_base;

  /// Specialized constructor.
  covmatrix(double variance):
    detail::covmatrix_base<1>::covmatrix_base({ variance })
  {}

  /// Correlation coefficient as a scalar.
  double corrcoef() const { return 1; };

  /// Volatility as a scalar;
  double volatility() const { return std::sqrt((*this)(0,0)); };
};


/// Covariance matrix in terms of volatilities and correlations (2x2).
template<>
class covmatrix<2>: public detail::covmatrix_base<2>
{
 public:
  using detail::covmatrix_base<2>::covmatrix_base;
  
  /// Specialized constructor.
  static covmatrix<2> from_vols_and_corr(
      double vol1, double vol2, double corrcoef
  ){
    return detail::covmatrix_base<2>({
      {vol1*vol1, vol1*vol2*corrcoef},
      {vol1*vol2*corrcoef, vol2*vol2}
    });
  }

  /// Correlation coefficient as a scalar.
  double corrcoef() const { return get_correlations()(0, 1); };
};


} // namespace math
} // namespace fpricing

#endif /* TYPES_HPP */
