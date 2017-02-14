// fdmoments.hpp - computing moments via finite-difference approximations.

#ifndef FDMOMENTS_HPP
#define FDMOMENTS_HPP 

#include <fpricing/math/types.hpp>


namespace fpricing {

namespace math {

namespace detail {

/// Central approximation for the first derivative.
template<class F>
decltype(auto) fd_first(const F& f, double x, double dx)
{
  return (
      f(x + dx) - f(x - dx)
  ) / (2*dx);
}

/// Central approximation for the second derivative.
template<class F, class ValType>
decltype(auto) fd_second(const F& f, double x, double dx, ValType f_at_x)
{
  return (
      f(x + dx) - 2*f_at_x + f(x - dx)
  ) / (dx * dx);
}

/// Central approximation for a mixed second derivative.
template<class F>
decltype(auto) fd_mixed(const F& f, double x, double y,
                        double dx, double dy)
{
  return (
      f(x + dx, y + dy) - f(x + dx, y - dy) + 
      - f(x - dx, y + dy) + f(x - dx, y - dy)
  ) / (4 * dx * dy);
}

} // namespace detail

/// The mean of a random vector given its Laplace tranform.
template<int N, class Lt>
math::vector<N> fd_mean(const Lt& lt, double eps=1e-6)
{
  math::vector<N> mean;
  for (int i=0; i<N; ++i)
  {
    auto ei = math::cxvector<N>::Unit(i);
    auto chfi = [&lt, ei](double t){
      auto expr = math::komplex(0, t) * ei;
      return lt( expr.matrix() );
    };
    mean(i) = std::imag( detail::fd_first(chfi, 0, eps) );
  }
  return mean;
}

/// Cov matrix of a random vector with zero mean given its Laplace tranform.
template<int N, class Lt>
math::covmatrix<N> fd_cov(const Lt& lt, double eps=1e-6)
{
  math::covmatrix<N> cov = math::covmatrix<N>::Zero();
  for (int i=0; i<N; ++i)
  {
    auto ei = math::cxvector<N>::Unit(i);
    // variance terms
    auto chfi = [&lt, ei](double t){
      auto expr = math::komplex(0, t) * ei;
      return lt( expr.matrix() );
    };
    cov(i, i) =  -std::real(detail::fd_second(chfi, 0, eps, 1.0));
    // covariance terms
    for (int j=0; j<i; ++j)
    {
      auto ej = math::cxvector<N>::Unit(j);
      auto chfij = [&lt, ei, ej](double ti, double tj){
        auto expr = math::komplex(0, ti) * ei + math::komplex(0, tj) * ej;
        return lt (expr.matrix() );
      };
      cov(i, j) = -std::real(detail::fd_mixed(chfij, 0, 0, eps, eps));
      cov(j, i) = cov(i, j);
    }
  }
  return cov;
}

/// Cov matrix of a random vector with zero mean given its Laplace tranform.
template<int N, class Lt>
math::covmatrix<N> fd_cov(const Lt& lt, math::vector<N> mean, double eps=1e-6)
{
  return fd_cov<N, Lt>(lt, eps) - mean * mean.transpose();
}

} // namespace math

} // namespace fpricing

#endif /* FDMOMENTS_HPP */
