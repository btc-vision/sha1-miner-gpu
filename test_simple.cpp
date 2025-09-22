#include <sycl/sycl.hpp>
#include <iostream>

int main() {
    try {
        auto devices = sycl::device::get_devices(sycl::info::device_type::gpu);
        std::cout << "Found " << devices.size() << " GPU devices" << std::endl;
        for (size_t i = 0; i < devices.size(); ++i) {
            std::cout << "Device " << i << ": " << devices[i].get_info<sycl::info::device::name>() << std::endl;
        }
        return 0;
    } catch (const std::exception& e) {
        std::cout << "Exception: " << e.what() << std::endl;
        return 1;
    }
}