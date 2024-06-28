#include "depth_sensor.h"

DepthSensor::DepthSensor() {
    prop.setZero(53);
    latent = torch::zeros({1, 32}, torch::kFloat);
    yaw = torch::zeros({2}, torch::kFloat);
};

DepthSensor::~DepthSensor() {
    this->stop();
}

void DepthSensor::start() {
    send_thread = std::thread(std::bind(&DepthSensor::send_prop, this));
    recv_thread = std::thread(std::bind(&DepthSensor::recv_latent_yaw, this));
}

void DepthSensor::stop() {
    if (send_thread.joinable()) {
        send_thread.join();
    }
    if (recv_thread.joinable()) {
        recv_thread.join();
    }
}

void DepthSensor::send_prop() {
    int sockfd;
    struct sockaddr_in server_addr;
    if ((sockfd = socket(AF_INET, SOCK_DGRAM, 0)) < 0) {
        std::cerr << "Socket creation error\n";
    }

    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(12345);
    server_addr.sin_addr.s_addr = inet_addr("192.168.1.103");

    while (true) {
        {
            std::vector<char> byteArray;
            const std::lock_guard<std::mutex> lock(prop_mutex);
            int dataSize = prop.size() * sizeof(float);
            byteArray.resize(dataSize);
            std::memcpy(byteArray.data(), prop.data(), dataSize);
            sendto(sockfd, byteArray.data(), byteArray.size(), 0, (struct sockaddr*)&server_addr, sizeof(server_addr));
            // std::cout << "Tensor sent\n" << prop << std::endl;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    close(sockfd);
}

void DepthSensor::get_prop(const VecXf& new_prop) {
    const std::lock_guard<std::mutex> lock(prop_mutex);
    prop = new_prop;
    // std::cout << "Policy set prop\n" << prop << std::endl;
}

void DepthSensor::recv_latent_yaw() {
    int sockfd;
    struct sockaddr_in server_addr;
    if ((sockfd = socket(AF_INET, SOCK_DGRAM, 0)) < 0) {
        std::cerr << "Socket creation error\n";
    }

    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(12345);
    server_addr.sin_addr.s_addr = INADDR_ANY;

    if (bind(sockfd, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        std::cerr << "Bind failed\n";
        close(sockfd);
    }

    while (true) {
        char buffer[1024 * 1024];  // Adjust buffer size as needed
        struct sockaddr_in client_addr;
        socklen_t addr_len = sizeof(client_addr);

        int num_bytes = recvfrom(sockfd, buffer, sizeof(buffer), 0, (struct sockaddr*)&client_addr, &addr_len);
        if (num_bytes < 0) {
            std::cerr << "Receive failed\n";
            exit(EXIT_FAILURE);
        }

        float* data_ptr = reinterpret_cast<float*>(buffer);
        int num_elements = num_bytes / sizeof(float);
        auto options = torch::TensorOptions().dtype(torch::kFloat32);
        torch::Tensor latent_yaw = torch::from_blob(data_ptr, {num_elements}, options).clone();
        
        {
            const std::lock_guard<std::mutex> lock(latent_mutex);
            latent = latent_yaw.slice(0, 0, 32).view({1, 32});
            // std::cout << "Received latent\n" << latent << std::endl;
        }

        {
            const std::lock_guard<std::mutex> lock(yaw_mutex);
            yaw = latent_yaw.slice(0, -2, latent_yaw.size(0));
            // std::cout << "Received yaw\n" << yaw << std::endl;
            yaw = yaw * 1.5;
        }
    }

    close(sockfd);
}

torch::Tensor DepthSensor::get_latent() {
    const std::lock_guard<std::mutex> lock(latent_mutex);
    // std::cout << "Policy get latent\n" << latent << std::endl;
    return latent;
}

torch::Tensor DepthSensor::get_yaw() {
    const std::lock_guard<std::mutex> lock(yaw_mutex);
    // std::cout << "Policy get yaw\n" << yaw << std::endl;
    return yaw;
}

