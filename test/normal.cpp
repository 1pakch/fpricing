#define BOOST_TEST_MODULE normal

#include <boost/test/included/unit_test.hpp>
#include <fpricing/math/normal.hpp>


using namespace fpricing;


BOOST_AUTO_TEST_CASE(init)
{
  auto mu = Eigen::Vector2d({0, 0});
  auto S = Eigen::Matrix2d({{1, 0}, {0, 1}});
  auto d = math::distr::Normal<2>(mu, S);
}
