#include <iostream>
#include <fstream>
#include <camera/device_discovery.h>

int main() {
    std::cout << ">>> Starting Insta360 SDK minimal test" << std::endl;

    ins_camera::DeviceDiscovery discovery;
    auto list = discovery.GetAvailableDevices();

    std::cout << ">>> Found " << list.size() << " camera(s)" << std::endl;

    for (const auto& device : list) {
        std::cout << "Serial: " << device.serial_number
                  << " | Name: " << device.camera_name
                  << " | FW: " << device.fw_version << std::endl;
    }

    return 0;
}
