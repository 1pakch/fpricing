#define BOOST_TEST_MODULE ndspreadcall

#include <boost/test/included/unit_test.hpp>

#include <fpricing/processes/gbm.hpp>
#include <fpricing/pricers/spread/ndspreadcall.hpp>


using namespace fpricing;


double bb76spreadCall(double F1, double F2, double strike,
                      double v1, double v2, double rho,
                      double tau)
{
  auto process = BivariateBlack76Process(v1, v2, rho);
  auto state = math::vector<2>({std::log(F1), std::log(F2)});
  auto distribution = process.log_state_distr(0, tau, state);
  return ndspreadcall(distribution, strike);
}


// test values obtained using simplemodels/black76spread commit 7ca2600
BOOST_AUTO_TEST_CASE(bb76pricing)
{
    BOOST_CHECK_CLOSE(0.041131,
        bb76spreadCall( 10,   7,  5, 0.2, 0.2, 0.50, 0.5),
        0.01);

    BOOST_CHECK_CLOSE(0.002343,
        bb76spreadCall( 10,   7,  5, 0.2, 0.2, 0.99, 1.0),
        0.01);

    BOOST_CHECK_CLOSE(0.256690,
        bb76spreadCall(110, 100, 10, 0.3, 0.3, 0.90, 0.002),
        0.01);
}
