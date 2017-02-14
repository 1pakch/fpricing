#define BOOST_TEST_MODULE normal

#include <iostream>

#include <boost/test/included/unit_test.hpp>
#include <fpricing/math/types.hpp>
#include <fpricing/math/normal.hpp>
#include <fpricing/math/fdmoments.hpp>


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


BOOST_AUTO_TEST_CASE(moments)
{
  const int N = 3;
  const double step = 1e-4;
  const double eps_mean = 1e-2;
  const double eps_cov = 1e-1;
  
  auto mu = vector<N>({1, 2, 3});
  auto cov = covmatrix<N>({
      {1, 1, 2},
      {1, 4, 4},
      {2, 4, 16}}
  );
  auto d = distr::Normal<N>(mu, cov);

  auto lt = [&d](cxvector<N> z){ return d.laplace_transform(z); };

  auto mu_approx = fd_mean<N>(lt, step);
  //std::cout << mu_approx << std::endl;
  
  BOOST_CHECK_CLOSE( mu(0), mu_approx(0), eps_mean );
  BOOST_CHECK_CLOSE( mu(1), mu_approx(1), eps_mean );

  auto cov_approx = fd_cov<N>(lt, mu, step);
  //std::cout << cov_approx << std::endl;
  
  BOOST_CHECK_CLOSE( cov(0,0), cov_approx(0,0), eps_cov );
  BOOST_CHECK_CLOSE( cov(1,1), cov_approx(1,1), eps_cov );
  BOOST_CHECK_CLOSE( cov(2,2), cov_approx(2,2), eps_cov );
  
  BOOST_CHECK_CLOSE( cov(0,1), cov_approx(0,1), eps_cov );
  BOOST_CHECK_CLOSE( cov(0,2), cov_approx(0,2), eps_cov );
  BOOST_CHECK_CLOSE( cov(1,2), cov_approx(1,2), eps_cov );
}
