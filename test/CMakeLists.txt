include_directories (${PROJECT_SOURCE_DIR}/include)

add_executable(gauss_legendre
	gauss_legendre.cpp)

add_executable(bb76spreadcall
	bb76spreadcall.cpp)

target_link_libraries(
	gauss_legendre
	fpricing
	${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})

target_link_libraries(
	bb76spreadcall
	fpricing
	${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})
