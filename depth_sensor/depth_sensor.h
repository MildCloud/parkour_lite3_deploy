#ifndef DEPTH_SENSOR_H_
#define DEPTH_SENSOR_H_

#include "common_types.h"
#include "torch/torch.h"
#include <arpa/inet.h>
using namespace types;

class DepthSensor {

    public:
        DepthSensor();
        ~DepthSensor();
        void start();
        void stop();
        void get_prop(const VecXf& new_prop);
        torch::Tensor get_latent();
        torch::Tensor get_yaw();

    private:
        VecXf prop;
        torch::Tensor latent;
        torch::Tensor yaw;
        std::mutex prop_mutex;
        std::mutex latent_mutex;
        std::mutex yaw_mutex;
        std::thread send_thread;
        std::thread recv_thread;
        void send_prop();
        void recv_latent_yaw();
};

#endif