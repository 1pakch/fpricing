// gauss_legendre_weights.cpp - constants for Gauss-Legendre integration
//
// The constants are declared as constexprs in the header file
//
//     fpricing/math/gauss_legendre.hpp  
//
// but are defined (allocated) here. Executables or libraries using
// the hedaer above integration should link against the object file
// resulting from this file.


#include <fpricing/math/gauss_legendre.hpp>


namespace fpricing {

namespace math {

namespace detail {

constexpr double gauss_legendre_rule<8>::weights[];
constexpr double gauss_legendre_rule<8>::points[];

constexpr double gauss_legendre_rule<16>::weights[];
constexpr double gauss_legendre_rule<16>::points[];

constexpr double gauss_legendre_rule<24>::weights[];
constexpr double gauss_legendre_rule<24>::points[];

constexpr double gauss_legendre_rule<32>::weights[];
constexpr double gauss_legendre_rule<32>::points[];

constexpr double gauss_legendre_rule<48>::weights[];
constexpr double gauss_legendre_rule<48>::points[];

} // namespace detail

} // namespace math

} // namespace fpricing 2
