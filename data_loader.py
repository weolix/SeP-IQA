import os
import json
import random
import pandas as pd
import torch
import ffmpeg
from decord import VideoReader, cpu
from PIL import Image

def uniform_sample(l, n, randomize=True):
    """
    均匀采样函数。
    从列表 l 中均匀采样 n 个元素。
    """
    gap = len(l) / n
    if randomize:
        idxs = [int(i * gap + random.uniform(0, gap)) for i in range(n)]
        idxs = [min(i, len(l) - 1) for i in idxs]
    else:
        # uniform sampling
        idxs = [int(i * gap + gap / 2) for i in range(n)]
    return [l[i] for i in idxs]


def encode_video(video_path, num_frames=16, random_sampling=True):
    """
    视频编码函数。
    读取视频文件并采样指定数量的帧。
    """
    vr = VideoReader(video_path, ctx=cpu(0))
    base_fps_divisor = 6
    if random_sampling:
        sample_fps_divisor = base_fps_divisor + random.uniform(-1, 1)
        sample_fps = max(1, round(vr.get_avg_fps() / sample_fps_divisor))
    else:
        sample_fps = round(vr.get_avg_fps() / base_fps_divisor)
    
    frame_idx = [i for i in range(0, len(vr), sample_fps)]
    if len(frame_idx) > num_frames:
        frame_idx = uniform_sample(frame_idx, num_frames, randomize=random_sampling)
    
    frames = vr.get_batch(frame_idx).numpy()
    frames = [Image.fromarray(v.astype('uint8')) for v in frames]
    return frames

class video_dataset(torch.utils.data.Dataset):
    """
    视频数据集类。
    读取视频文件并进行采样，用于视频质量评估。
    """
    def __init__(self, anno_file, data_prefix, phase, sample_types):
        super().__init__()

        self.video_infos = []
        self.phase = phase
        self.sample_types = sample_types
        self.mean = torch.FloatTensor([123.675, 116.28, 103.53])
        self.std = torch.FloatTensor([58.395, 57.12, 57.375])
        self.samplers = {}

        with open(anno_file, "r") as fin:
            for line in fin:
                line_split = line.strip().split(",")
                filename, a, t, label = line_split
                label = float(a), float(t), float(label)

                filename = os.path.join(data_prefix, filename)
                self.video_infos.append(dict(filename=filename, label=label))

    def __len__(self):
        return len(self.video_infos)

    def __getitem__(self, idx):
        video = self.video_infos[idx]
        video_path = video["filename"]

        metadata = ffmpeg.probe(video_path)
        meta_stream = metadata["streams"][0]

        video_inf = {
            "resolution": f"{meta_stream['width']}x{meta_stream['height']}",
            "frame rate": f"{eval(meta_stream['avg_frame_rate']):.2f}fps",
            "bit rate": f"{int(meta_stream['bit_rate'])//1000}Kbps",
            "codec": meta_stream["codec_name"],
        }

        a, t, video_label = video["label"]
        video_frames = encode_video(video_path, num_frames=self.sample_types["clip_len"])
        video = {"info": video_inf, "data": video_frames}
        return video, video_label
    
    def collate_fn(self, batch):
        videos, labels = zip(*batch)
        videos = [video for video in videos]
        labels = torch.tensor(labels, dtype=torch.float32)
        return videos, labels
    

class ImageJsonDataset(torch.utils.data.Dataset):
    """
    图像数据集类 (JSON 格式标注)。
    读取 JSON 格式的标注文件，用于图像质量评估。
    """
    def __init__(self, dir, anno_file):
        self.dir = dir

        with open(anno_file, 'r') as f:
            self.data = json.load(f)["files"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.dir + "/" + self.data[idx]['image']
        label = self.data[idx]['score']
        image = Image.open(img_path).convert('RGB')
        width, height = image.size
        file_format = os.path.basename(img_path).split(".")[-1].upper()
        image_inf = {
            "Format": image.format if image.format else file_format,
            "File Size": f"{os.path.getsize(img_path)>>10:.0f}KB",
            "Resolution": f"{width}x{height}",
            "img_name": os.path.basename(img_path),
        }
        img = {"info": image_inf, "data": image}
        return img, label
    
    def collate_fn(self, batch):
        images, labels = zip(*batch)
        images = [img for img in images]
        labels = torch.tensor(labels, dtype=torch.float32)
        return images, labels
    

class ImageCsvDataset(torch.utils.data.Dataset):
    """
    图像数据集类 (CSV 格式标注)。
    读取 CSV 格式的标注文件，用于图像质量评估。
    """
    def __init__(self, dir, anno_file, image_key, score_key):
        super().__init__()
        self.dir = dir
        # 用pandas读取csv文件
        df = pd.read_csv(anno_file)
        self.data = df[[image_key, score_key]].values.tolist()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.dir, self.data[idx][0])
        label = self.data[idx][1]
        image = Image.open(img_path).convert('RGB')
        width, height = image.size
        file_format = os.path.basename(img_path).split(".")[-1].upper()
        image_inf = {
            "name": os.path.basename(img_path),
            "Format": image.format if image.format else file_format,
            "File Size": f"{os.path.getsize(img_path)>>10:.0f}KB",
            "Resolution": f"{width}x{height}",
        }
        img = {"info": image_inf, "data": image}
        return img, label
    
    def collate_fn(self, batch):
        images, labels = zip(*batch)
        images = [img for img in images]
        labels = torch.tensor(labels, dtype=torch.float32)
        return images, labels


class TopkDataset(torch.utils.data.Dataset):
    """
    Top-K 数据集类。
    用于加载预先计算好的 Top-K logits 和 indices，用于快速评估或拟合。
    """
    def __init__(self, topk_data):
        super().__init__()
        self.topk_data = topk_data

    def __len__(self):
        return len(self.topk_data["logits"])

    def __getitem__(self, idx):
        logits = self.topk_data["logits"][idx]
        indices = self.topk_data["indices"][idx]
        gt_score = self.topk_data["gt_scores"][idx]
        return logits, indices, gt_score
    
    def collate_fn(self, batch):
        logits, indices, gt_scores = zip(*batch)
        logits = torch.stack(logits, dim=0)
        indices = torch.stack(indices, dim=0)
        gt_scores = torch.tensor(gt_scores, dtype=torch.float32)
        return logits, indices, gt_scores

def list_collate(batch):
    # batch is a list of tuples
    n = len(batch[0])
    data = []
    for i in range(n):
        data.append([batch[j][i] for j in range(len(batch))])
    data.append(torch.FloatTensor(data.pop()))
    return data
