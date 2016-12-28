// test_gauss_legendre.cpp - Gauss-Legendre integration tests.

#define BOOST_TEST_MODULE gauss_legendre
#include <boost/test/included/unit_test.hpp>
#include <cmath>

#include <fpricing/math/gauss_legendre.hpp>

static const double Pi = std::acos(-1);

// Integrate a univariate Gaussian
BOOST_AUTO_TEST_CASE(gaussian)
{
  auto f = [](double x){ return std::exp(-(x*x)/2 ); };
  BOOST_CHECK_CLOSE(
      std::sqrt(2 * Pi),
      fpricing::math::gauss_legendre<32>(f, -4, 4),
      0.01
  );
}
