cmake_minimum_required(VERSION 3.2)
project(automy-basic-opencl)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(automy_basic_opencl SHARED
	src/Context.cpp
	src/Kernel.cpp
	src/Program.cpp
)
add_library(automy_basic_opencl_static STATIC
	src/Context.cpp
	src/Kernel.cpp
	src/Program.cpp
)

target_include_directories(automy_basic_opencl
	PUBLIC include
)
target_include_directories(automy_basic_opencl_static
	PUBLIC include
)

target_link_libraries(automy_basic_opencl
	OpenCL::OpenCL
)
target_link_libraries(automy_basic_opencl_static
	OpenCL::OpenCL
)

target_compile_definitions(automy_basic_opencl PUBLIC NOGDI)
target_compile_definitions(automy_basic_opencl_static PUBLIC NOGDI)

if(MSVC)
	include(GenerateExportHeader)
	GENERATE_EXPORT_HEADER(automy_basic_opencl)

	set_target_properties(automy_basic_opencl PROPERTIES ENABLE_EXPORTS ON)
	set_target_properties(automy_basic_opencl PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)

	target_include_directories(automy_basic_opencl
		PUBLIC ${CMAKE_CURRENT_BINARY_DIR}
	)
	target_include_directories(automy_basic_opencl_static
		PUBLIC ${CMAKE_CURRENT_BINARY_DIR}
	)
endif()

install(DIRECTORY kernel/ DESTINATION share/kernel)
install(DIRECTORY include/ DESTINATION include)

install(TARGETS automy_basic_opencl DESTINATION lib)
install(TARGETS automy_basic_opencl_static DESTINATION lib)
