import socket
import cv2
import numpy as np
import torch
from torch import nn
import pyrealsense2 as rs
from datetime import datetime
import time
import sys
import pickle

# wifi 871
# sensor board nx ipv4 192.168.1.103


class RecurrentDepthBackbone(nn.Module):
    def __init__(self, base_backbone, env_cfg) -> None:
        super().__init__()
        activation = nn.ELU()
        last_activation = nn.Tanh()
        self.base_backbone = base_backbone
        if env_cfg == None:
            self.combination_mlp = nn.Sequential(
                                    nn.Linear(32 + 53, 128),
                                    activation,
                                    nn.Linear(128, 32)
                                )
        else:
            self.combination_mlp = nn.Sequential(
                                        nn.Linear(32 + env_cfg.env.n_proprio, 128),
                                        activation,
                                        nn.Linear(128, 32)
                                    )
        self.rnn = nn.GRU(input_size=32, hidden_size=512, batch_first=True)
        self.output_mlp = nn.Sequential(
                                nn.Linear(512, 32+2),
                                last_activation
                            )
        self.hidden_states = None

    def forward(self, depth_image, proprioception):
        depth_image = self.base_backbone(depth_image)
        depth_latent = self.combination_mlp(torch.cat((depth_image, proprioception), dim=-1))
        # depth_latent = self.base_backbone(depth_image)
        depth_latent, self.hidden_states = self.rnn(depth_latent[:, None, :], self.hidden_states)
        depth_latent = self.output_mlp(depth_latent.squeeze(1))
        
        return depth_latent

    def detach_hidden_states(self):
        self.hidden_states = self.hidden_states.detach().clone()

class DepthOnlyFCBackbone58x87(nn.Module):
    def __init__(self, prop_dim, scandots_output_dim, hidden_state_dim, output_activation=None, num_frames=1):
        super().__init__()

        self.num_frames = num_frames
        activation = nn.ELU()
        self.image_compression = nn.Sequential(
            # [1, 58, 87]
            nn.Conv2d(in_channels=self.num_frames, out_channels=32, kernel_size=5),
            # [32, 54, 83]
            nn.MaxPool2d(kernel_size=2, stride=2),
            # [32, 27, 41]
            activation,
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            activation,
            nn.Flatten(),
            # [32, 25, 39]
            nn.Linear(64 * 25 * 39, 128),
            activation,
            nn.Linear(128, scandots_output_dim)
        )

        if output_activation == "tanh":
            self.output_activation = nn.Tanh()
        else:
            self.output_activation = activation

    def forward(self, images: torch.Tensor):
        images_compressed = self.image_compression(images.unsqueeze(1))
        latent = self.output_activation(images_compressed)

        return latent


depth_backbone = DepthOnlyFCBackbone58x87(53, 32, 512)
model = RecurrentDepthBackbone(depth_backbone, None)
load_path = "/home/ysc/tianshu/model_9000.pt"

def load_model():
    global model
    try:
        if torch.cuda.is_available():
            device = 'cuda'
            print("Using CUDA")
        else:
            device = 'cpu'
            print("Using CPU")
        ac_state_dict = torch.load(load_path, map_location=device)
        model.load_state_dict(ac_state_dict['depth_encoder_state_dict'])
        model.to('cuda')
        model.eval()
    except RuntimeError as e:
        print("Error loading the model.")
        sys.exit(1)

def depth_encoder_infer(resized_depth_img, prop_tensor):
    img_tensor = torch.from_numpy(resized_depth_img.astype(np.float32)).clone()
    img_tensor = img_tensor.unsqueeze(0)
    prop_tensor = prop_tensor.unsqueeze(0)
    if torch.cuda.is_available():
        img_tensor = img_tensor.to('cuda')
        prop_tensor = prop_tensor.to('cuda')
    with torch.no_grad():
        output = model.forward(img_tensor, prop_tensor).squeeze().cpu()
    return output

def send_depth_data():
    load_model()
    resized_depth_image = np.ones((58, 87))
    prop_tensor = torch.zeros(53, dtype=torch.float)
    output_tensor = depth_encoder_infer(resized_depth_image, prop_tensor)
    print('warm up', output_tensor)

    sock_recv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_recv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock_recv.bind(('192.168.1.103', 12345))
    sock_send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    pipeline = rs.pipeline()
    pipeline.start()

    frame_count = 0

    while True:
        data, address = sock_recv.recvfrom(1024)
        prop_tensor = torch.from_numpy(np.frombuffer(data, dtype=np.float32))
        print('proprioception', prop_tensor)

        frames = pipeline.wait_for_frames()  # Wait for a new frame
        depth = frames.get_depth_frame()  # Get the depth frame

        # Convert the depth frame to a NumPy array
        depth_image = np.asanyarray(depth.get_data(), dtype=np.uint16)

        # Convert depth from millimeters to meters
        depth_image = depth_image.astype(np.float32) / 1000
        
        # Resize the depth image using OpenCV
        depth_image = cv2.resize(depth_image, (106, 60), interpolation=cv2.INTER_LINEAR)
        
        # Crop image
        depth_image = depth_image[:-2, 4:-4]

        # Clip the values between 0 and 2 meters
        np.clip(depth_image, 0, 2, out=depth_image)

        depth_image = cv2.resize(depth_image, (87, 58), interpolation=cv2.INTER_LINEAR)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        depth_image = cv2.morphologyEx(depth_image, cv2.MORPH_CLOSE, kernel)
        for i in range(len(depth_image)):
            if np.all(depth_image[i] == 2):
                depth_image[:i][:] = 2
                break

        depth_image_normalized = ((depth_image - np.min(depth_image)) / (np.max(depth_image) - np.min(depth_image)) * 255).astype(np.uint8)
        cv2.imwrite('./depth_images/depth_image_'+ str(frame_count) +'.png', depth_image_normalized)
        frame_count += 1

        depth_image = depth_image / 2  - 0.5
        # print('depth_image', depth_image)
        # print('depth_image.shape', depth_image.shape)

        output_tensor = depth_encoder_infer(depth_image, prop_tensor)
        
        print(f"Output: {output_tensor}")

        output = output_tensor.numpy().tobytes()
        sock_send.sendto(output, ('192.168.1.120', 12345))


def main():
    send_depth_data()

if __name__ == "__main__":
    main()

