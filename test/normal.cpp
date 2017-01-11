#define BOOST_TEST_MODULE normal

#include <boost/test/included/unit_test.hpp>
#include <fpricing/math/types.hpp>
#include <fpricing/math/normal.hpp>


using namespace fpricing;
using namespace fpricing::math;


BOOST_AUTO_TEST_CASE(init)
{
  auto mu = vector<2>({1, 2});
  auto S = covmatrix<2>({{1, 1}, {1, 4}});
  auto d = distr::Normal<2>(mu, S);

  const double eps = 1e-8;

  auto a1 = d.premultiply(matrix<1, 2>({{1, 0}}));
  BOOST_CHECK_EQUAL( a1.ndim(), 1 );
  BOOST_CHECK_CLOSE( a1.mean(0), 1, eps );
  BOOST_CHECK_CLOSE( a1.cov(0,0), 1, eps );

  auto a2 = d.premultiply(matrix<1, 2>({{0, 1}}));
  BOOST_CHECK_EQUAL( a2.ndim(), 1 );
  BOOST_CHECK_CLOSE( a2.mean(0), 2, eps );
  BOOST_CHECK_CLOSE( a2.cov(0,0), 4, eps );

  auto sum = d.premultiply(matrix<1, 2>({{1, 1}}));
  BOOST_CHECK_EQUAL( sum.ndim(), 1 );
  BOOST_CHECK_CLOSE( sum.mean(0), 3, eps );
  BOOST_CHECK_CLOSE( sum.cov(0,0), 7, eps );
}
