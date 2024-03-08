/*
 * Kernel.cpp
 *
 *  Created on: Jul 12, 2018
 *      Author: mad
 */

#include <automy/basic_opencl/Kernel.h>


namespace automy {
namespace basic_opencl {

Kernel::Kernel(cl_kernel kernel_, bool with_arg_map)
	:	kernel(kernel_)
{
	size_t length = 0;
	if(cl_int err = clGetKernelInfo(kernel, CL_KERNEL_FUNCTION_NAME, 0, 0, &length)) {
		throw opencl_error_t("clGetKernelInfo(CL_KERNEL_FUNCTION_NAME) failed with " + get_error_string(err));
	}
	if(!length) {
		throw std::runtime_error("kernel name too short");
	}
	name.resize(length);
	if(cl_int err = clGetKernelInfo(kernel, CL_KERNEL_FUNCTION_NAME, name.size(), &name[0], &length)) {
		throw opencl_error_t("clGetKernelInfo(CL_KERNEL_FUNCTION_NAME) failed with " + get_error_string(err));
	}
	if(length) {
		name.resize(length - 1);
	}
	
	if(with_arg_map) {
		cl_uint num_args = 0;
		if(cl_int err = clGetKernelInfo(kernel, CL_KERNEL_NUM_ARGS, sizeof(num_args), &num_args, &length)) {
			throw opencl_error_t("clGetKernelInfo(CL_KERNEL_NUM_ARGS) failed with " + get_error_string(err));
		}

		for(cl_uint i = 0; i < num_args; ++i) {
			if(cl_int err = clGetKernelArgInfo(kernel, i, CL_KERNEL_ARG_NAME, 0, 0, &length)) {
				throw opencl_error_t("clGetKernelArgInfo(CL_KERNEL_ARG_NAME, 0, 0) failed with " + get_error_string(err));
			}

			// workaround for API issue observed with ROCK 5B, Mali-G610 MP4 GPU, OpenCL implementation
			// CL_KERNEL_ARG_NAME, param_value_size=0, param_value=NULL, always 0 in param_value_size_ret
			// if detected, try fallback with big enough temp buffer, to find size
			if(!length) {
				char paramv[256];
				if(cl_int err = clGetKernelArgInfo(kernel, i, CL_KERNEL_ARG_NAME, sizeof(paramv), &paramv, &length)) {
					throw opencl_error_t("clGetKernelArgInfo(CL_KERNEL_ARG_NAME, tmpsize, tmpbuf) failed with " + get_error_string(err));
				}
			}

			if(!length) {
				throw std::runtime_error("kernel argument name too short");
			}
			std::string arg;
			arg.resize(length);
			if(cl_int err = clGetKernelArgInfo(kernel, i, CL_KERNEL_ARG_NAME, arg.size(), &arg[0], &length)) {
				throw opencl_error_t("clGetKernelArgInfo(CL_KERNEL_ARG_NAME) failed with " + get_error_string(err));
			}
			if(length) {
				arg.resize(length - 1);
			}
			arg_list.push_back(arg);
			arg_map[arg] = i;
		}
	}
}

Kernel::~Kernel() {
	if(kernel) {
		clReleaseKernel(kernel);
	}
}

std::shared_ptr<Kernel> Kernel::create(cl_kernel kernel, bool with_arg_map) {
	return std::make_shared<Kernel>(kernel, with_arg_map);
}

void Kernel::set_local(const std::string& arg, const size_t& num_bytes) {
	auto it = arg_map.find(arg);
	if(it != arg_map.end()) {
		if(clSetKernelArg(kernel, it->second, num_bytes, 0)) {
			throw opencl_error_t("clSetKernelArg() failed for " + name + " : " + arg);
		}
	} else {
		throw std::logic_error("no such argument '" + arg + "' in kernel '" + name + "'");
	}
}

void Kernel::enqueue(std::shared_ptr<CommandQueue> queue, const size_t& global_size) {
	if(cl_int err = clEnqueueNDRangeKernel(queue->get(), kernel, 1, 0, &global_size, 0, 0, 0, 0)) {
		throw opencl_error_t("clEnqueueNDRangeKernel() failed for kernel '" + name + "' with " + get_error_string(err));
	}
}

void Kernel::enqueue(std::shared_ptr<CommandQueue> queue, const size_t& global_size, const size_t& local_size) {
	if(cl_int err = clEnqueueNDRangeKernel(queue->get(), kernel, 1, 0, &global_size, &local_size, 0, 0, 0)) {
		throw opencl_error_t("clEnqueueNDRangeKernel() failed for kernel '" + name + "' with " + get_error_string(err));
	}
}

void Kernel::enqueue_ceiled(std::shared_ptr<CommandQueue> queue, const size_t& global_size, const size_t& local_size) {
	const auto global_size_ = global_size + (local_size - (global_size % local_size)) % local_size;
	enqueue(queue, global_size_, local_size);
}

void Kernel::enqueue_2D(std::shared_ptr<CommandQueue> queue, const std::array<size_t, 2>& global_size) {
	if(cl_int err = clEnqueueNDRangeKernel(queue->get(), kernel, 2, 0, global_size.data(), 0, 0, 0, 0)) {
		throw opencl_error_t("clEnqueueNDRangeKernel() failed for kernel '" + name + "' with " + get_error_string(err));
	}
}

void Kernel::enqueue_2D(std::shared_ptr<CommandQueue> queue, const std::array<size_t, 2>& global_size, const std::array<size_t, 2>& local_size) {
	if(cl_int err = clEnqueueNDRangeKernel(queue->get(), kernel, 2, 0, global_size.data(), local_size.data(), 0, 0, 0)) {
		throw opencl_error_t("clEnqueueNDRangeKernel() failed for kernel '" + name + "' with " + get_error_string(err));
	}
}

void Kernel::enqueue_ceiled_2D(std::shared_ptr<CommandQueue> queue, const std::array<size_t, 2>& global_size, const std::array<size_t, 2>& local_size) {
	std::array<size_t, 2> global_size_ = global_size;
	for(int i = 0; i < 2; ++i) {
		global_size_[i] += (local_size[i] - (global_size[i] % local_size[i])) % local_size[i];
	}
	enqueue_2D(queue, global_size_, local_size);
}

void Kernel::enqueue_3D(std::shared_ptr<CommandQueue> queue, const std::array<size_t, 3>& global_size) {
	if(cl_int err = clEnqueueNDRangeKernel(queue->get(), kernel, 3, 0, global_size.data(), 0, 0, 0, 0)) {
		throw opencl_error_t("clEnqueueNDRangeKernel() failed for kernel '" + name + "' with " + get_error_string(err));
	}
}

void Kernel::enqueue_3D(std::shared_ptr<CommandQueue> queue, const std::array<size_t, 3>& global_size, const std::array<size_t, 3>& local_size) {
	if(cl_int err = clEnqueueNDRangeKernel(queue->get(), kernel, 3, 0, global_size.data(), local_size.data(), 0, 0, 0)) {
		throw opencl_error_t("clEnqueueNDRangeKernel() failed for kernel '" + name + "' with " + get_error_string(err));
	}
}

void Kernel::enqueue_ceiled_3D(std::shared_ptr<CommandQueue> queue, const std::array<size_t, 3>& global_size, const std::array<size_t, 3>& local_size) {
	std::array<size_t, 3> global_size_ = global_size;
	for(int i = 0; i < 3; ++i) {
		global_size_[i] += (local_size[i] - (global_size[i] % local_size[i])) % local_size[i];
	}
	enqueue_3D(queue, global_size_, local_size);
}

void Kernel::print_info(std::ostream& out) {
	out << name << "(";
	for(size_t i = 0; i < arg_list.size(); ++i) {
		if(i > 0) {
			out << ", ";
		}
		out << arg_list[i];
	}
	out << ")" << std::endl;
}


} // basic_opencl
} // automy
