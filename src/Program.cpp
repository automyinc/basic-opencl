/*
 * Program.cpp
 *
 *  Created on: Jul 12, 2018
 *      Author: mad
 */

#include <automy/basic_opencl/Program.h>

#include <map>
#include <set>
#include <mutex>
#include <fstream>


namespace automy {
namespace basic_opencl {

std::shared_ptr<Program> Program::create(cl_context context) {
	return std::make_shared<Program>(context);
}

Program::Program(cl_context context)
	:	context(context)
{
}

Program::~Program() {
	if(program) {
		clReleaseProgram(program);
	}
}

void Program::add_include_path(const std::string& path) {
	includes.insert(path);
}

void Program::add_source(const std::string& file_name)
{
	std::vector<std::string> source_dirs {""};
	source_dirs.insert(source_dirs.end(), includes.begin(), includes.end());

	for(const auto& dir : source_dirs) {
		std::ifstream in(dir + file_name);
		if(in.good()) {
			sources.push_back(std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()));
			return;
		}
	}
	throw std::runtime_error("no such file: '" + file_name + "'");
}

void Program::add_source_code(const std::string& source)
{
	sources.push_back(source);
}

void Program::create_from_source() {
	if(program) {
		throw std::logic_error("program already created");
	}
	
	std::vector<const char*> list;
	for(const std::string& source : sources) {
		list.push_back(source.c_str());
	}
	
	cl_int err = 0;
	program = clCreateProgramWithSource(context, list.size(), &list[0], 0, &err);
	if(err) {
		throw opencl_error_t("clCreateProgramWithSource() failed with " + get_error_string(err));
	}
}

bool Program::build(const std::vector<cl_device_id>& devices, bool with_arg_names)
{
	if(!program) {
		throw std::logic_error("program == nullptr");
	}
	have_arg_info = with_arg_names;
	
	std::string options_ = options;
	if(with_arg_names) {
		options_ += " -cl-kernel-arg-info";
	}
	for(const auto& path : includes) {
		if(!path.empty()) {
			options_ += " -I " + path;
		}
	}
	
	bool success = true;
	if(cl_int err = clBuildProgram(program, devices.size(), devices.data(), options_.c_str(), 0, 0)) {
		if(err != CL_BUILD_PROGRAM_FAILURE) {
			throw opencl_error_t("clBuildProgram() failed with " + get_error_string(err));
		}
		success = false;
	}
	
	for(cl_device_id device : devices) {
		size_t length = 0;
		cl_build_status status;
		if(cl_int err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_STATUS, sizeof(status), &status, &length)) {
			throw opencl_error_t("clGetProgramBuildInfo(CL_PROGRAM_BUILD_STATUS) failed with " + get_error_string(err));
		}
		if(status != CL_BUILD_SUCCESS) {
			success = false;
		}
		if(cl_int err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, 0, &length)) {
			throw opencl_error_t("clGetProgramBuildInfo(CL_PROGRAM_BUILD_LOG, 0, 0) failed with " + get_error_string(err));
		}
		if(length > 0) {
			std::string log;
			log.resize(length);
			if(cl_int err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log.size(), &log[0], &length)) {
				throw opencl_error_t("clGetProgramBuildInfo(CL_PROGRAM_BUILD_LOG) failed with " + get_error_string(err));
			}
			if(length) {
				log.resize(length - 1);
			}
			build_log.push_back(log);
		}
	}
	return success;
}

void Program::print_sources(std::ostream& out) const {
	for(const std::string& source : sources) {
		out << source << std::endl;
	}
}

void Program::print_build_log(std::ostream& out) const {
	for(const std::string& log : build_log) {
		out << log << std::endl;
	}
}

std::shared_ptr<Kernel> Program::create_kernel(const std::string& name) const {
	cl_int err = 0;
	cl_kernel kernel = clCreateKernel(program, name.c_str(), &err);
	if(err) {
		throw opencl_error_t("clCreateKernel() failed for '" + name + "' with " + get_error_string(err));
	}
	return Kernel::create(kernel, have_arg_info);
}


} // basic_opencl
} // automy
