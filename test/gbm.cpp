#define BOOST_TEST_MODULE gbm

#include <iostream>
#include <cmath>
#include <boost/test/included/unit_test.hpp>

#include <fpricing/processes/gbm.hpp>
#include <fpricing/math/fdmoments.hpp>


using namespace fpricing;
using namespace fpricing::math;


BOOST_AUTO_TEST_CASE(moments)
{
  const double cov = 1.0;
  const double tau = 0.25;
  auto process = Black76Process(cov);

  const double step = 1e-6;
  
  // test that the process is a martingale
  {
    double spot = std::log(100);
    auto spotv = vector<1>({ spot });
    
    auto lt = [&process, tau, spotv](const cxvector<1> z){
        return process.transform_coefficients(0, tau, z).expdot(spotv);
    };


    auto mu_approx = fd_mean<1>(lt, step);
    std::cout << mu_approx << std::endl;
    BOOST_CHECK_CLOSE(
        std::exp(spot),
        std::real(lt(cxvector<1>({1.0}))),
        1.0
    );
  }

  // test the covariance of returns
  {
    auto lt = [&process, tau](const cxvector<1> z){
      return std::exp(
          process.transform_coefficients(0, tau, z).const_term
      );
    };

    auto cov_approx = fd_cov<1>(lt, step);
    std::cout << cov_approx << std::endl;

    BOOST_CHECK_CLOSE( cov*tau, cov_approx(0,0), 25 );
  }
}
