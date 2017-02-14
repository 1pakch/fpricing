#define BOOST_TEST_MODULE b76vanillacall

#include <boost/test/included/unit_test.hpp>

#include <fpricing/processes/gbm.hpp>
#include <fpricing/pricers/vanilla/gbs.hpp>
#include <iostream>

using namespace fpricing;


double atmApprox(double F0, double vol, double tau)
{
  return 0.4 * F0 * vol * std::sqrt(tau);
}

double b76call(double F0, double strike, double vol, double tau)
{
  auto process = Black76Process(vol);
  auto state = math::vector<1>({std::log(F0)});
  auto distribution = process.log_state_distr(0, tau, state);
  return gbs(distribution, strike);
}

BOOST_AUTO_TEST_CASE(bb76pricing)
{
  BOOST_CHECK_CLOSE(
      7.9655674554058038,
      b76call(100, 100, 0.2, 1),
      1e-6);  
  
  BOOST_CHECK_CLOSE(
      b76call(1, 1, 0.9, 1),
      atmApprox(1, 0.9, 1),
      5); 
 
  BOOST_CHECK_CLOSE(
      b76call(1, 1, 0.1, 0.1),
      atmApprox(1, 0.1, 0.1),
      5);  
}
