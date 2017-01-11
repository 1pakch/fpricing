// gaussian.hpp - Gaussian processes and their conditional distributions.

#ifndef GAUSSIAN_HPP
#define GAUSSIAN_HPP 

#include <fpricing/math/types.hpp>
#include <fpricing/math/normal.hpp>


namespace fpricing {

//
// Gaussian process class
//

/// A Gaussian process with a customizable drift specification.
template<int N, template<int> typename Drift>
class GaussianProcess
{
 public:
 
  Drift<N> drift {};

  const math::covmatrix<N> cov {};


  GaussianProcess() = default;

  math::distr::Normal<N> get_conditional_distribution(
      double t,
      double T,
      math::vector<N> state
  ){
    return drift.get_conditional_distribution(t, T, cov, state);
  } 

};


//
// Different drift specifications
//

/// Makes the exponent of the process a martingale.
template<int N>
class ExponentialMartingaleDrift
{
 public:

  ExponentialMartingaleDrift() = default;
 
 protected:

  friend class GaussianProcess<N, ExponentialMartingaleDrift>;

  decltype(auto) get_conditional_distribution(
      double t,
      double T,
      math::covmatrix<N> cov,
      math::vector<N> starting_state
  ){
    return math::distr::Normal<N>(
        starting_state - 0.5 * (T-t) * cov.diagonal(),
        (T-t)*cov
    );
  }
};


/// Constant drift specification for Gaussian processes
template<int N>
class ConstantDrift
{
 public:
 
  const math::vector<N> constant_term {};

 protected:

  friend class GaussianProcess<N, ConstantDrift>;
  
  decltype(auto) get_conditional_distribution(
      double t,
      double T,
      math::covmatrix<N> cov,
      math::vector<N> starting_state
  ){
    return math::distr::Normal<N>(
        (T-t) * constant_term + starting_state,
        (T-t) * cov
    );
  }
};

/*
/// Affine drift specification for Gaussian processes
template<int N>
class AffineDrift
{

 public:

  const math::vector<N> constant_term {};
  
  const math::matrix<N, N> linear_term {};
 
 protected:

  friend class GaussianProcess<N, AffineDrift>;
  
  decltype(auto) get_conditional_distribution(
      double t,
      double T,
      math::covmatrix<N> cov,
      math::vector<N> starting_state
  ){
    NotImplementedTemplate;
  }
};
*/

//
// Specific Gaussian processes
//

/// Univariate Black76 process.
struct Black76Process:
    public GaussianProcess<1, ExponentialMartingaleDrift>
{
  Black76Process(double vol):
    GaussianProcess<1, ExponentialMartingaleDrift>{
      {},
      math::covmatrix<1>({vol*vol})
    }
  {}
};

/// Two correlated Black-76 processes.
struct BivariateBlack76Process:
    public GaussianProcess<2, ExponentialMartingaleDrift>
{
  BivariateBlack76Process(double vol1, double vol2, double corr):
    GaussianProcess<2, ExponentialMartingaleDrift>{
      {}, // zero drift
      math::covmatrix<2>::from_vols_and_corr(vol1, vol2, corr)
    }
  {}
};

} // namespace fpricing


#endif /* GAUSSIAN_HPP */
