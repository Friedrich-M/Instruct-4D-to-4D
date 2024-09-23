import decord
decord.bridge.set_bridge('torch')

from torch.utils.data import Dataset
import os
from einops import rearrange
from PIL import Image
import torch
from torchvision import transforms


class VideoDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        prompt: str,
        width: int = 512,
        height: int = 512,
        n_sample_frames: int = 24,
        sample_start_idx: int = 0,
        sample_frame_rate: int = 1,
    ):
        self.data_path = data_path
        self.prompt = prompt
        self.prompt_ids = None

        self.width = width
        self.height = height
        self.n_sample_frames = n_sample_frames
        self.sample_start_idx = sample_start_idx
        self.sample_frame_rate = sample_frame_rate
        
    def __len__(self):
        return 1

    def __getitem__(self, index):
        # load and sample video frames
        vr = decord.VideoReader(self.data_path, width=self.width, height=self.height)
        sample_index = list(range(self.sample_start_idx, len(vr), self.sample_frame_rate))[:self.n_sample_frames]
        video = vr.get_batch(sample_index) # (f, h, w, c)
        video = rearrange(video, "f h w c -> f c h w")

        example = {
            "pixel_values": (video / 127.5 - 1.0),
            "prompt_ids": self.prompt_ids
        }

        return example
    
    
class MultiFrameDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        prompt: str,
        width: int = 256,
        height: int = 512,
        n_sample_frames: int = 24,
        sample_start_idx: int = 0,
    ):
        self.data_path = data_path
        self.prompt = prompt
        self.prompt_ids = None
                
        cam_index = [0]
        cams_list = sorted(os.listdir(data_path))
        cams_list = [cams_list[i] for i in cam_index]
        self.cams_path = [os.path.join(data_path, cam) for cam in cams_list]
        
        self.width = width
        self.height = height
        self.n_sample_frames = n_sample_frames
        self.sample_start_idx = sample_start_idx
        
        self.trans = transforms.Compose([
            transforms.Resize((self.height, self.width)),
            transforms.ToTensor(),
        ])
        
    def __len__(self):
        return len(self.cams_path)

    def __getitem__(self, index):
        frames_dir = self.cams_path[index]
        frames_list = sorted(os.listdir(frames_dir))
        frames_path = [os.path.join(frames_dir, frame) for frame in frames_list]
        
        frames_path = frames_path[self.sample_start_idx:self.sample_start_idx+self.n_sample_frames]
         
        # load and sample video frames
        frames = []
        for frame_path in frames_path:
            frame = Image.open(frame_path).convert('RGB')
            frame = self.trans(frame) # (c, h, w)
            frames.append(frame)
            
        frames = torch.stack(frames, dim=0) # (f, c, h, w)
        
        example = {
            "pixel_values": frames * 2 - 1,
            "prompt_ids": self.prompt_ids
        }
        
        return example
        
class MultiViewDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        prompt: str,
        width: int = 256,
        height: int = 512,
    ):
        self.data_path = data_path
        self.prompt = prompt
        self.prompt_ids = None
                
        cams_list = sorted(os.listdir(data_path))[-10:]
        self.cams_path = [os.path.join(data_path, cam) for cam in cams_list]
        
        self.width = width
        self.height = height
        
        self.trans = transforms.Compose([
            transforms.Resize((self.height, self.width)) if (self.height != -1 and self.width != -1) else transforms.Lambda(lambda img: img),
            transforms.ToTensor(),
        ])
        
    def __len__(self):
        return len(self.cams_path)

    def __getitem__(self, index):
        frames_path = self.cams_path
         
        # load and sample video frames
        frames = []
        for frame_path in frames_path:
            frame = Image.open(frame_path).convert('RGB')
            frame = self.trans(frame) # (c, h, w)
            frames.append(frame)
            
        frames = torch.stack(frames, dim=0) # (f, c, h, w)
        
        example = {
            "pixel_values": frames * 2 - 1,
            "prompt_ids": self.prompt_ids
        }
        
        return example