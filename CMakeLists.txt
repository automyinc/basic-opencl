cmake_minimum_required(VERSION 3.2)
project(automy-basic-opencl)
add_compile_options(-DNOGDI)

file(GLOB GENERATED_SOURCE_FILES generated/src/*.cpp)

add_library(automy_basic_opencl SHARED
	${GENERATED_SOURCE_FILES}
	src/Context.cpp
	src/Kernel.cpp
	src/Program.cpp
)

target_include_directories(automy_basic_opencl
	PUBLIC include
)


include(GenerateExportHeader)
GENERATE_EXPORT_HEADER(automy_basic_opencl)
include_directories("${CMAKE_CURRENT_BINARY_DIR}")
set(AUTOMY_BASIC_OPENCL_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}" CACHE STRING "")	


if(MSVC)
	set_target_properties(automy_basic_opencl PROPERTIES ENABLE_EXPORTS 1)
	set_target_properties(automy_basic_opencl PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS 1)
	
	find_library(OpenCL_LIBRARY OpenCL HINTS ${OpenCL_LIBRARIES})
	add_library(OpenCL INTERFACE IMPORTED)
	set_target_properties(OpenCL PROPERTIES INTERFACE_LINK_LIBRARIES "${OpenCL_LIBRARY}")
	
	target_include_directories(automy_basic_opencl PRIVATE ${OpenCL_INCLUDE_DIR})		
endif()


target_link_libraries(automy_basic_opencl
	OpenCL
)

install(DIRECTORY kernel/ DESTINATION kernel)
install(DIRECTORY include/ DESTINATION include)

install(TARGETS automy_basic_opencl DESTINATION lib)
