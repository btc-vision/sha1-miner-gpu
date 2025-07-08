#include <CL/cl.h>
#include <iostream>
#include <vector>

int main() {
    cl_uint num_platforms;
    clGetPlatformIDs(0, nullptr, &num_platforms);
    std::cout << "Found " << num_platforms << " OpenCL platforms\n";
    
    std::vector<cl_platform_id> platforms(num_platforms);
    clGetPlatformIDs(num_platforms, platforms.data(), nullptr);
    
    for (cl_uint i = 0; i < num_platforms; i++) {
        char name[128];
        clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, sizeof(name), name, nullptr);
        std::cout << "\nPlatform " << i << ": " << name << "\n";
        
        cl_uint num_devices;
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices);
        std::vector<cl_device_id> devices(num_devices);
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, num_devices, devices.data(), nullptr);
        
        for (cl_uint j = 0; j < num_devices; j++) {
            clGetDeviceInfo(devices[j], CL_DEVICE_NAME, sizeof(name), name, nullptr);
            std::cout << "  Device " << j << ": " << name << "\n";
        }
    }
    return 0;
}
