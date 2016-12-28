include_directories (${PROJECT_SOURCE_DIR}/include)

add_executable(gauss_legendre
	gauss_legendre.cpp)

target_link_libraries(
	gauss_legendre
	fpricing
	${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})

