"""
mPLUG-Owl3 Zero-shot Video/Image Quality Assessment Script
此脚本用于使用 mPLUG-Owl3 模型进行零样本视频和图像质量评估。
包含模型定义、数据加载、多种评估方法（Prompt评估、Embedding评估、Q-Align/Q-Bench评估）以及工具函数。
"""

import os
import sys
import json
import time
import random
import glob
import yaml
import tqdm
import pandas as pd
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import transformers
import ffmpeg
from decord import VideoReader, cpu
from scipy.stats import spearmanr, pearsonr

# 本地模块
import dataset

# 环境变量设置
os.environ["AV_LOG_FORCE_NOCOLOR"] = "1"  # 去除颜色编码
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["AV_LOG_LEVEL"] = "quiet"
os.environ["FFMPEG_LOGLEVEL"] = "quiet"

# 设备配置
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device: ", device)

# TensorBoard 设置
date_str = time.strftime("%Y-%m-%d", time.localtime())
time_str = time.strftime("%H:%M:%S", time.localtime())
run_dir = f"runs/{date_str}/{time_str}"
writer = SummaryWriter(run_dir)


class MultimodalQualityEvaluator(nn.Module):
    """
    多模态质量评估器模型封装类。
    基于 mPLUG-Owl3 模型，支持 IQA (图像质量评估), VQA (视频质量评估), IAA (图像美学评估) 等任务。
    """
    def __init__(
        self,
        task="IQA",
        model_path="iic/mPLUG-Owl3-7B-241101",
    ):
        super().__init__()
        self.task = task
        
        # 加载配置
        config = transformers.AutoConfig.from_pretrained(
            model_path, trust_remote_code=True
        )

        # 加载模型
        self.LLM = transformers.AutoModel.from_pretrained(
            model_path,
            config=config,
            trust_remote_code=True,
            local_files_only=True,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
        )

        # 加载分词器和处理器
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
        self.LLMprocessor = self.LLM.init_processor(self.tokenizer)

    def forward(self, image_or_video=None, labels=None, **args):
        """
        前向传播函数。
        
        Args:
            image_or_video: 输入的图像或视频数据列表
            labels: 标签（可选）
            
        Returns:
            outputs: 模型的输出 logits
        """
        # 处理输入数据
        batched_inputs = self.processor(image_or_video)

        # 获取图像嵌入
        with torch.no_grad():
            image_embeds = self.LLM.forward_image(batched_inputs.pop("pixel_values"))
        
        # 语言模型前向传播
        outputs = self.LLM.language_model(
            image_embeds=image_embeds,
            labels=labels,
            **batched_inputs,
        )
        
        # 清理内存
        del batched_inputs, image_embeds
        torch.cuda.empty_cache()
    
        return outputs

    def processor(self, data_and_info):
        """
        数据预处理函数。
        将原始的图像/视频数据和信息转换为模型可接受的输入格式 (input_ids, attention_mask, pixel_values 等)。
        
        Args:
            data_and_info: 包含 "data" (图像/视频对象) 和 "info" (元数据) 的字典列表
            
        Returns:
            batched_inputs: 批处理后的输入字典
        """
        image_or_video = [d["data"] for d in data_and_info]
        media_type = "images" if isinstance(image_or_video[0], Image.Image) else "videos"
        batch_size = len(image_or_video)
        media_token = {"images": "<|image|>", "videos": "<|video|>"}

        # 1. 准备 Prompt 消息
        batched_messages = []
        for i in range(batch_size):
            if self.task == "VQA":
                messages = [
                    {
                        "role": "user",
                        "content": f"{media_token[media_type]}Taking into account the content and fluency of the {media_type[:-1]}, how would you rate the quality of this {media_type[:-1]}?",
                    },
                    {"role": "assistant", "content": "The quality of the image is very"},
                ]
            elif self.task == "IQA":
                messages = [
                    {
                        "role": "user",
                        "content": f"{media_token[media_type]}Taking into account the details and the rationality of the {media_type[:-1]}, how would you rate the quality of this {media_type[:-1]}?",
                    },
                    {"role": "assistant", "content": "The quality of the image is very"},
                ]
            elif self.task == "IAA":
                messages = [
                    {
                        "role": "user",
                        "content": f"{media_token[media_type]}Considering its artistic composition, color harmony, and overall visual appeal, use an adjective to describe the aesthetic quality of this image?",
                    },
                    {"role": "assistant", "content": "The image is"},
                ]
            else:
                # 默认 Prompt
                messages = [
                    {
                        "role": "user",
                        "content": f"{media_token[media_type]}Describe this {media_type[:-1]}.",
                    },
                    {"role": "assistant", "content": ""},
                ]

            batched_messages.append(messages)

        # 2. 逐个样本处理 (Tokenization & Image Processing)
        processed_outputs = []
        all_pixel_values = [] 

        for i in range(batch_size):
            current_messages = batched_messages[i]
            current_media = [image_or_video[i]] 

            single_process_dict = {
                "messages": current_messages,
                media_type: current_media, 
                "preface": True
            }

            # 调用 LLMprocessor
            single_output = self.LLMprocessor(**single_process_dict, return_tensors="pt")
            processed_outputs.append(single_output)
            
            # 收集 pixel_values
            if "pixel_values" in single_output:
                all_pixel_values.append(single_output["pixel_values"].to(self.dev))
            elif "pixel_values_videos" in single_output: 
                all_pixel_values.append(single_output["pixel_values_videos"].to(self.dev))
            else:
                print(f"Warning: 'pixel_values' not found for sample {i}")

        if not processed_outputs or not all_pixel_values:
             raise ValueError("No samples were processed successfully.")

        # 提取 input_ids 和 media_offset
        all_input_ids = [out['input_ids'].squeeze(0) for out in processed_outputs] 
        all_media_offsets = [out['media_offset'][0] for out in processed_outputs] 

        # 3. Padding (左填充)
        max_len = max(len(ids) for ids in all_input_ids)
        padded_input_ids = []
        attention_masks = []
        
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
            if pad_token_id is None:
                 raise ValueError("Tokenizer must have a pad_token_id or eos_token_id.")

        for input_ids in all_input_ids:
            current_len = len(input_ids)
            padding_len = max_len - current_len

            # 创建填充张量
            padding_tensor = torch.full((padding_len,), pad_token_id, dtype=input_ids.dtype, device=self.dev)
            # 左填充
            padded_ids = torch.cat([padding_tensor, input_ids.to(self.dev)], dim=0)
            padded_input_ids.append(padded_ids)

            # 创建 attention_mask (0: padding, 1: token)
            mask = torch.cat([
                torch.zeros(padding_len, dtype=torch.long, device=self.dev),
                torch.ones(current_len, dtype=torch.long, device=self.dev)
            ], dim=0)
            attention_masks.append(mask)

        # 堆叠成 Batch
        batched_input_ids = torch.stack(padded_input_ids, dim=0)
        batched_attention_mask = torch.stack(attention_masks, dim=0)

        # 4. 合并 Pixel Values
        try:
            batched_pixel_values = torch.cat(all_pixel_values, dim=0)
        except RuntimeError as e:
             print(f"Error concatenating pixel_values: {e}. Check if samples have consistent frame counts.")
             raise e

        # 5. 组装最终输入
        batched_inputs = {
            "input_ids": batched_input_ids,
            "attention_mask": batched_attention_mask,
            "media_offset": all_media_offsets, # 保持列表形式
            "pixel_values": batched_pixel_values,
        }

        return batched_inputs
  
    @property
    def dev(self):
        return self.LLM.device


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


def plcc_loss(y_pred, y):
    """
    PLCC (Pearson Linear Correlation Coefficient) 损失函数。
    结合了 MSE 损失和相关性损失。
    """
    sigma_hat, m_hat = torch.std_mean(y_pred, unbiased=False)
    y_pred = (y_pred - m_hat) / (sigma_hat + 1e-8)
    sigma, m = torch.std_mean(y, unbiased=False)
    y = (y - m) / (sigma + 1e-8)
    loss0 = torch.nn.functional.mse_loss(y_pred, y) / 4
    rho = torch.mean(y_pred * y)
    loss1 = torch.nn.functional.mse_loss(rho * y_pred, y) / 4
    return ((loss0 + loss1) / 2).float()


def embed_fit(model, val_dataset, val_embed, bsz=8, data_path=None):
    """
    Embedding 拟合函数。
    (实验性功能) 尝试微调模型或 embedding 以更好地拟合数据。
    """
    model.eval()
    if data_path is not None and os.path.exists(data_path):
        topk_data = torch.load(data_path)
        topk_dataset = TopkDataset(topk_data)
        valdataloader = DataLoader(topk_dataset, batch_size=bsz*5, shuffle=False, collate_fn=topk_dataset.collate_fn)
    else:
        valdataloader = DataLoader(val_dataset, batch_size=bsz, shuffle=False, num_workers=4, collate_fn=dataset.list_collate)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    val_gt_scores = []
    topk_data_logits = []
    topk_data_indices = []
    for i, batch in enumerate(tqdm.tqdm(valdataloader, desc=f"fitting", ncols=100)):
        k = 100

        if data_path is None or not os.path.exists(data_path):

            image_or_video = batch[0]
            labels = batch[1]

            with torch.no_grad():
                outputs = model(image_or_video=image_or_video)
                logits = outputs.logits

            last_token_logits = logits[:, -1, :] # Shape: (batch_size, vocab_size)

            topk = torch.topk(last_token_logits, k, dim=-1) # Shapes: (batch_size, k)
            top_k_logits, top_k_indices = topk.values, topk.indices
        
            topk_data_logits.append(top_k_logits.cpu())
            topk_data_indices.append(top_k_indices.cpu())

        else:
            top_k_logits, top_k_indices, labels = batch


        embedding_layer = model.LLM.get_input_embeddings()
        top_k_embeddings = embedding_layer(top_k_indices.to(model.LLM.device))
        val_embed_unsqueezed = val_embed.to(top_k_embeddings.device).unsqueeze(0).unsqueeze(0)

        weights = F.cosine_similarity(top_k_embeddings, val_embed_unsqueezed, dim=-1)
        weighted_logits = top_k_logits.to(model.LLM.device) * weights # Shape: (batch_size, k)
        score = torch.sum(weighted_logits, dim=-1) # Shape: (batch_size,)

        loss = plcc_loss(score, labels.to(score.device))
        loss.backward(retain_graph=True)
        optimizer.step()
        optimizer.zero_grad()
        
        val_gt_scores.extend(labels.cpu().tolist())

    if len(topk_data_logits) > 0:
        topk_data = {
            "logits": torch.cat(topk_data_logits, dim=0),
            "indices": torch.cat(topk_data_indices, dim=0),
            "gt_scores": torch.tensor(val_gt_scores),
        }

        return val_embed, topk_data 
    
    return val_embed, {}


def prompt_evaluate(model, val_dataset, val_embed, prompts_list, bsz=2, collect_bsz=None):
    """
    Prompt 评估函数。
    评估不同 Prompt 对模型性能的影响。
    
    Args:
        model: 模型实例
        val_dataset: 验证数据集
        val_embed: 用于评分的嵌入向量 (Anchor Embedding)
        prompts_list: 待评估的 Prompt 列表，每个元素包含 name, user_content, assistant_content
        bsz: 推理时的 batch size
        collect_bsz: 收集数据时的 batch size，如果为 None 则使用 bsz
    
    Returns:
        dict: 每个 Prompt 的评估结果 (SRCC, PLCC 等)
    """
    import gc
    
    model.eval()
    
    # 如果没有指定 collect_bsz，使用更小的 batch size 进行数据收集以节省显存
    if collect_bsz is None:
        collect_bsz = max(1, bsz // 2)

    collect_dataloader = DataLoader(val_dataset, batch_size=collect_bsz, shuffle=False, num_workers=2, collate_fn=val_dataset.collate_fn)

    all_prompt_results = {}
    
    # 1. 预先收集真实标签 (Ground Truth) 和数据索引
    # 避免在每次 Prompt 评估时都重新加载整个数据集
    val_gt_scores = []
    data_indices = [] 
    
    print(f"Collecting ground truth labels and indices (batch_size={collect_bsz})...")
    
    try:
        for i, batch in enumerate(tqdm.tqdm(collect_dataloader, desc="Collecting data", ncols=100)):
            labels = batch[1]
            val_gt_scores.extend(labels.cpu().tolist())
            
            # 记录当前 batch 的数据索引
            start_idx = i * collect_bsz
            end_idx = start_idx + len(labels)
            data_indices.extend(list(range(start_idx, end_idx)))
            
            # 定期清理内存
            if i % 50 == 0:
                torch.cuda.empty_cache()
                gc.collect()
                
    except Exception as e:
        print(f"Error during data collection: {e}")
        raise e
    
    val_gt_scores = torch.tensor(val_gt_scores)
    print(f"Total samples collected: {len(val_gt_scores)}")
    
    # 释放 collect_dataloader
    del collect_dataloader
    torch.cuda.empty_cache()
    gc.collect()
    
    # 2. 对每个 Prompt 进行评估
    for prompt_idx, prompt_info in enumerate(prompts_list):
        prompt_name = prompt_info["name"]
        user_content = prompt_info["user_content"]
        assistant_content = prompt_info["assistant_content"]
        
        print(f"\n========== Evaluating Prompt {prompt_idx+1}/{len(prompts_list)}: {prompt_name} ==========")
        print(f"User content: {user_content}")
        print(f"Assistant content: {assistant_content}")
        
        # 定义临时的 processor 函数，用于注入当前 Prompt
        def temp_processor(data_and_info):
            image_or_video = [d["data"] for d in data_and_info]
            media_type = "images" if isinstance(image_or_video[0], Image.Image) else "videos"
            batch_size = len(image_or_video)
            media_token = {"images": "<|image|>", "videos": "<|video|>"}

            # 使用当前 Prompt 构建消息
            batched_messages = []
            for i in range(batch_size):
                messages = [
                    {
                        "role": "user",
                        "content": f"{media_token[media_type]}{user_content}",
                    },
                    {"role": "assistant", "content": assistant_content},
                ]
                batched_messages.append(messages)

            # --- 以下逻辑与原 processor 相同 ---
            processed_outputs = []
            all_pixel_values = []

            for i in range(batch_size):
                current_messages = batched_messages[i]
                current_media = [image_or_video[i]]

                single_process_dict = {
                    "messages": current_messages,
                    media_type: current_media,
                    "preface": True
                }

                single_output = model.LLMprocessor(**single_process_dict, return_tensors="pt")
                processed_outputs.append(single_output)
                
                if "pixel_values" in single_output:
                    all_pixel_values.append(single_output["pixel_values"].to(model.dev))
                elif "pixel_values_videos" in single_output:
                    all_pixel_values.append(single_output["pixel_values_videos"].to(model.dev))
                else:
                    print("Warning: 'pixel_values' not found")

            if not processed_outputs or not all_pixel_values:
                raise ValueError("No samples were processed successfully.")

            all_input_ids = [out['input_ids'].squeeze(0) for out in processed_outputs]
            all_media_offsets = [out['media_offset'][0] for out in processed_outputs]

            max_len = max(len(ids) for ids in all_input_ids)

            padded_input_ids = []
            attention_masks = []
            pad_token_id = model.tokenizer.pad_token_id
            if pad_token_id is None:
                pad_token_id = model.tokenizer.eos_token_id

            for input_ids in all_input_ids:
                current_len = len(input_ids)
                padding_len = max_len - current_len

                padding_tensor = torch.full((padding_len,), pad_token_id, dtype=input_ids.dtype, device=model.dev)
                padded_ids = torch.cat([padding_tensor, input_ids.to(model.dev)], dim=0)
                padded_input_ids.append(padded_ids)

                mask = torch.cat([
                    torch.zeros(padding_len, dtype=torch.long, device=model.dev),
                    torch.ones(current_len, dtype=torch.long, device=model.dev)
                ], dim=0)
                attention_masks.append(mask)

            batched_input_ids = torch.stack(padded_input_ids, dim=0)
            batched_attention_mask = torch.stack(attention_masks, dim=0)
            batched_pixel_values = torch.cat(all_pixel_values, dim=0)

            batched_inputs = {
                "input_ids": batched_input_ids,
                "attention_mask": batched_attention_mask,
                "media_offset": all_media_offsets,
                "pixel_values": batched_pixel_values,
            }

            return batched_inputs
        
        # 临时替换 model 的 processor
        original_processor = model.processor
        model.processor = temp_processor
        
        val_pred_scores = []
        
        try:
            with torch.no_grad():
                # 创建新的 dataloader 进行推理
                inference_dataloader = DataLoader(val_dataset, batch_size=bsz, shuffle=False, num_workers=2, collate_fn=val_dataset.collate_fn)
                
                for i, batch in enumerate(tqdm.tqdm(inference_dataloader, desc=f"Evaluating {prompt_name}", ncols=100)):
                    image_or_video = batch[0]
                    
                    outputs = model(image_or_video=image_or_video)
                    logits = outputs.logits
                    
                    # 获取最后一个 token 的 logits
                    last_token_logits = logits[:, -1, :]
                    
                    # 计算 Top-K 加权分数
                    k = 100
                    topk = torch.topk(last_token_logits, k, dim=-1)
                    top_k_logits, top_k_indices = topk.values, topk.indices
                    
                    embedding_layer = model.LLM.get_input_embeddings()
                    top_k_embeddings = embedding_layer(top_k_indices)
                    val_embed_unsqueezed = val_embed.to(top_k_embeddings.device).unsqueeze(0).unsqueeze(0)
                    
                    weights = F.cosine_similarity(top_k_embeddings, val_embed_unsqueezed, dim=-1)
                    weighted_logits = top_k_logits.to(weights.device) * weights
                    score = torch.sum(weighted_logits, dim=-1)
                    
                    val_pred_scores.extend(score.cpu().tolist())
                    
                    # 定期清理内存
                    if i % 20 == 0:
                        torch.cuda.empty_cache()
                        gc.collect()
                
                # 释放 dataloader
                del inference_dataloader
                torch.cuda.empty_cache()
                gc.collect()
        
        except Exception as e:
            print(f"Error during evaluation of {prompt_name}: {e}")
            # 恢复原来的 processor
            model.processor = original_processor
            continue
        
        # 恢复原来的 processor
        model.processor = original_processor
        
        # 计算相关系数
        val_pred_scores = torch.tensor(val_pred_scores)
        
        # 确保预测分数和真实分数长度一致
        if len(val_pred_scores) != len(val_gt_scores):
            min_len = min(len(val_pred_scores), len(val_gt_scores))
            val_pred_scores = val_pred_scores[:min_len]
            current_gt_scores = val_gt_scores[:min_len]
            print(f"Warning: Length mismatch, using first {min_len} samples")
        else:
            current_gt_scores = val_gt_scores
        
        spearmanrcc = spearmanr(val_pred_scores[:,0] if val_pred_scores.dim() > 1 else val_pred_scores, current_gt_scores)
        pearsonrcc = pearsonr(val_pred_scores[:,0] if val_pred_scores.dim() > 1 else val_pred_scores, current_gt_scores)
        
        # 存储结果
        all_prompt_results[prompt_name] = {
            "srcc": float(spearmanrcc.statistic),
            "plcc": float(pearsonrcc.statistic),
            "user_content": user_content,
            "assistant_content": assistant_content,
            "pred_scores": val_pred_scores.tolist(),
            "gt_scores": current_gt_scores.tolist()
        }
        
        print(f"{prompt_name}: SRCC={spearmanrcc.statistic:.4f}, PLCC={pearsonrcc.statistic:.4f}")
        
        # 每个 Prompt 评估完后清理内存
        torch.cuda.empty_cache()
        gc.collect()
    
    return all_prompt_results




def evaluate_multiple_embeds(model, val_dataset, val_embeds, bsz=8):
    """
    多 Embedding 评估函数。
    评估不同的 Anchor Embedding 和不同的 Top-K 值对模型性能的影响。
    
    Args:
        model: 模型实例
        val_dataset: 验证数据集
        val_embeds: 待评估的 Embedding 列表 (字典形式，key为名称，value为tensor)
        bsz: 推理时的 batch size
    
    Returns:
        dict: 包含每个 Embedding 和每个 K 值的结果
    """
    model.eval()
    valdataloader = DataLoader(val_dataset, batch_size=bsz, shuffle=False, num_workers=4, collate_fn=dataset.list_collate)
    Ks = [100, 200, 300]  # 定义要测试的 K 值列表
    
    all_embed_results = {}
    
    with torch.no_grad():
        val_gt_scores = []
        all_logits = []  # 存储所有的 logits 用于后续不同 K 值的测试
        
        # 1. 收集所有数据的 Logits 和标签 (只需一次推理)
        print("Collecting logits and labels...")
        for i, batch in enumerate(tqdm.tqdm(valdataloader, desc=f"Collecting logits", ncols=100)):
            image_or_video = batch[0]
            labels = batch[1]

            outputs = model(image_or_video=image_or_video)
            logits = outputs.logits

            # 获取最后一个 token 的 logits
            last_token_logits = logits[:, -1, :]  # Shape: (batch_size, vocab_size)
            all_logits.append(last_token_logits.cpu())
            val_gt_scores.extend(labels.cpu().tolist())
        
        # 拼接所有 logits
        all_logits = torch.cat(all_logits, dim=0)  # Shape: (total_samples, vocab_size)
        val_gt_scores = torch.tensor(val_gt_scores)
        
        # 2. 对每个 Embedding 进行评估
        # val_embeds 应该是一个字典或者列表，这里假设是列表，如果需要名称可以改为字典
        for embed_idx, val_embed in enumerate(val_embeds):
            embed_name = f"embed_{embed_idx}"
            print(f"\n========== Evaluating Embed {embed_idx} ==========")
            embed_results = {}
            
            # 3. 对每个 K 值进行评估
            for K in Ks:
                print(f"Evaluating Embed {embed_idx} with K={K}")
                val_pred_scores = []
                
                # 分批处理 logits 以节省显存
                batch_size_eval = bsz * 4 
                num_batches = (len(all_logits) + batch_size_eval - 1) // batch_size_eval
                
                for batch_idx in tqdm.tqdm(range(num_batches), desc=f"Embed {embed_idx}, K={K}", ncols=100):
                    start_idx = batch_idx * batch_size_eval
                    end_idx = min((batch_idx + 1) * batch_size_eval, len(all_logits))
                    
                    batch_logits = all_logits[start_idx:end_idx].to(device)
                    
                    # 获取 Top-K Logits 和 Indices
                    topk_output = torch.topk(batch_logits, K, dim=-1)
                    top_k_logits = topk_output.values    # Shape: (batch_size, k)
                    top_k_indices = topk_output.indices  # Shape: (batch_size, k)

                    # 获取词嵌入
                    embedding_layer = model.LLM.get_input_embeddings()
                    top_k_embeddings = embedding_layer(top_k_indices)  # Shape: (batch_size, k, embedding_dim)

                    # 计算余弦相似度权重
                    val_embed_unsqueezed = val_embed.to(device).unsqueeze(0).unsqueeze(0)
                    weights = F.cosine_similarity(top_k_embeddings, val_embed_unsqueezed, dim=-1) # Shape: (batch_size, k)

                    # 加权求和
                    weighted_logits = top_k_logits * weights 
                    score = torch.sum(weighted_logits, dim=-1) 

                    val_pred_scores.extend(score.cpu().tolist())
                
                # 计算相关系数
                val_pred_scores = torch.tensor(val_pred_scores)
                spearmanrcc = spearmanr(val_pred_scores, val_gt_scores)
                pearsonrcc = pearsonr(val_pred_scores, val_gt_scores)
                
                embed_results[K] = {
                    "srcc": float(spearmanrcc.statistic),
                    "plcc": float(pearsonrcc.statistic),
                }
                
                print(f"Embed {embed_idx}, K={K}: SRCC={spearmanrcc.statistic:.4f}, PLCC={pearsonrcc.statistic:.4f}")
            
            all_embed_results[embed_name] = embed_results

    return all_embed_results


class NumpyEncoder(json.JSONEncoder):
    """
    Numpy JSON 编码器。
    用于解决 json.dump 无法直接序列化 numpy 数据类型的问题。
    """
    def default(self, obj):
        import numpy as np
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def q_evaluate(model, val_dataset, val_embed, bsz=2):
    """
    综合评估函数。
    同时评估 Embedding 方法 (New Method), Q-Bench 方法, 和 Q-Align 方法。
    
    Args:
        model: 模型实例
        val_dataset: 验证数据集
        val_embed: Embedding 方法使用的 Anchor Embedding
        bsz: batch size
        
    Returns:
        return_dict: 包含各方法的 SRCC 和 PLCC 结果
        exp_data: 包含各方法的预测分数和真实分数，用于后续分析
    """
    model.eval()
    
    # Q-Bench 和 Q-Align 的特定 Token ID
    qbench_tokens = [model.tokenizer.encode(t)[0] for t in ["good", "poor"]]
    qalign_tokens = [model.tokenizer.encode(t)[0] for t in ["bad", "poor", "fair", "good", "excellent"]]
    
    valdataloader = DataLoader(val_dataset, batch_size=bsz, shuffle=False, num_workers=0, collate_fn=dataset.list_collate)
    
    with torch.no_grad():
        val_pred_scores = []      # Embedding Method Scores
        qbench_val_scores = []    # Q-Bench Scores
        qalign_val_scores = []    # Q-Align Scores
        val_gt_scores = []        # Ground Truth Scores
        
        for i, batch in enumerate(tqdm.tqdm(valdataloader, desc=f"Validation", ncols=100)):

            image_or_video = batch[0]
            labels = batch[1]
            val_gt_scores.extend(labels.cpu().tolist())

            outputs = model(image_or_video=image_or_video)
            logits = outputs.logits
            last_token_logits = logits[:, -1, :] # Shape: (batch_size, vocab_size)
            
            # ---------------------- 1. Embedding Method (New Method) ----------------------
            k = 100
            topk = torch.topk(last_token_logits, k, dim=-1)
            top_k_logits, top_k_indices = topk.values, topk.indices

            embedding_layer = model.LLM.get_input_embeddings()
            top_k_embeddings = embedding_layer(top_k_indices)
            val_embed_unsqueezed = val_embed.to(top_k_embeddings.device).unsqueeze(0).unsqueeze(0)

            weights = F.cosine_similarity(top_k_embeddings, val_embed_unsqueezed, dim=-1)
            weighted_logits = top_k_logits.to(weights.device) * weights
            score = torch.sum(weighted_logits, dim=-1)

            val_pred_scores.extend(score.cpu().tolist())
            
            # ---------------------- 2. Q-Bench Method ----------------------
            # 计算 "good" 和 "poor" 的概率
            binary_logits = last_token_logits[:, qbench_tokens] # Shape: (batch_size, 2)
            binary_probality = torch.softmax(binary_logits, dim=-1)
            q_bench_score = binary_probality[:, 0] # 取 "good" 的概率作为分数

            qbench_val_scores.extend(q_bench_score.cpu().tolist())

            # ---------------------- 3. Q-Align Method ----------------------
            # 计算 5 个等级词的加权平均分
            target_logits = last_token_logits[:, qalign_tokens] # Shape: (batch_size, 5)
            target_probality = torch.softmax(target_logits, dim=-1)
            # 权重: bad=1, poor=2, fair=3, good=4, excellent=5
            target_scores = torch.sum(target_probality * torch.tensor([1, 2, 3, 4, 5], device=target_probality.device), dim=-1)

            qalign_val_scores.extend(target_scores.cpu().tolist())

        # 转换为 Tensor
        val_pred_scores = torch.tensor(val_pred_scores)
        qbench_val_scores = torch.tensor(qbench_val_scores)
        qalign_val_scores = torch.tensor(qalign_val_scores)
        val_gt_scores = torch.tensor(val_gt_scores)
        
        # 计算相关系数
        spearmanrcc = spearmanr(val_pred_scores, val_gt_scores).statistic
        pearsonrcc = pearsonr(val_pred_scores, val_gt_scores).statistic
        
        qbench_spearmanrcc = spearmanr(qbench_val_scores, val_gt_scores).statistic
        qbench_pearsonrcc = pearsonr(qbench_val_scores, val_gt_scores).statistic
        
        qalign_spearmanrcc = spearmanr(qalign_val_scores, val_gt_scores).statistic
        qalign_pearsonrcc = pearsonr(qalign_val_scores, val_gt_scores).statistic

        return_dict = {
            "srcc": spearmanrcc,
            "plcc": pearsonrcc,
            "qbench_srcc": qbench_spearmanrcc,
            "qbench_plcc": qbench_pearsonrcc,
            "qalign_srcc": qalign_spearmanrcc,
            "qalign_plcc": qalign_pearsonrcc,
        }

        exp_data = {
            "val_pred_scores": val_pred_scores.tolist(),
            "qbench_val_scores": qbench_val_scores.tolist(),
            "qalign_val_scores": qalign_val_scores.tolist(),
            "val_gt_scores": val_gt_scores.tolist(),
        }

    return return_dict, exp_data


def get_top_logits(img_dir, model, batch_size=1, topk=10):
    """
    Top-K Logits 分析工具。
    遍历指定目录下的图片，输出模型预测的 Top-K Logits 及其对应的词汇。
    用于定性分析模型对图像的理解。
    
    Args:
        img_dir: 图片目录路径
        model: 模型实例
        batch_size: 推理 batch size
        topk: 显示前 K 个结果
    """
    import glob

    # 支持常见图片格式
    img_paths = []
    for ext in ["jpg", "jpeg", "png", "bmp", "webp"]:
        img_paths.extend(glob.glob(os.path.join(img_dir, f"*.{ext}")))
    img_paths.sort()
    print(f"共找到 {len(img_paths)} 张图片")

    # 构造数据列表
    images = []
    for img_path in img_paths:
        image = Image.open(img_path).convert("RGB")
        width, height = image.size
        file_format = os.path.basename(img_path).split(".")[-1].upper()
        image_inf = {
            "Format": image.format if image.format else file_format,
            "File Size": f"{os.path.getsize(img_path)>>10:.0f}KB",
            "Resolution": f"{width}x{height}",
            "img_name": os.path.basename(img_path),
        }
        images.append({"info": image_inf, "data": image})

    # 分批推理
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        with torch.no_grad():
            outputs = model(image_or_video=batch)
            logits = outputs.logits  # shape: (batch, seq_len, vocab_size)
            
        for j, img in enumerate(batch):
            last_token_logits = logits[j, -1, :]  # 取最后一个 token 的 logits
            topk_vals, topk_ids = torch.topk(last_token_logits, topk)
            topk_words = model.tokenizer.batch_decode(topk_ids.unsqueeze(1))
            
            print(f"Image: {img['info']['img_name']}")
            print(f"Top-{topk} logits:")
            for idx, (val, tid, word) in enumerate(zip(topk_vals.tolist(), topk_ids.tolist(), topk_words)):
                print(f"  {idx+1:2d}: logit={val:.4f}  id={tid:<6d}  word='{word.strip()}'")
            print("-" * 40)


def get_embed(model, device, TASK="IQA"):
    """
    获取任务相关的 Anchor Embedding。
    根据预定义的正向和负向词汇，计算它们的 Embedding 差值向量。
    
    Args:
        model: 模型实例
        device: 设备
        TASK: 任务名称 (IQA, IAA, IQA1, IQA2)
        
    Returns:
        val_embed: 计算得到的 Anchor Embedding 向量
    """
    embeddings = {}
    text_dict = {   
        "IQA" : 
        {
            "positive" : " perfect superb outstanding excellent fantastic stunning striking phenomenal brilliant magnificent amazing remarkable beautiful awesome breathtaking great good decent fine sharp clear suitable vibrant rich vivid bright colorful",
            "negative" : " bad terrible awful poor poor horrible disappointing unacceptable inadequate deficient blurry fuzzy compromised chaotic distorted weak mediocre sub lacking unclear dark noisy low problematic insufficient"
        },
        "IAA" : 
        {
            "positive": " beautiful stunning enchanting harmonious artistic pleasing exquisite stunning elegant graceful balanced vibrant evocative poignant serene sublime picturesque appealing striking gorgeous charming delightful sophisticated",
            "negative": " mediocre poorly dull bland chaotic disple lacking amateur overly sub monotonous average clutter uninspired unpleasant discord garish mundane tacky glaring simplistic flat"
        },
        "IQA1":
        {
            "positive": " sharp clear crisp detailed vibrant excellent superb pristine flawless high-resolution stunning perfect refined polished exquisite brilliant outstanding magnificent impressive superior fine luxurious premium professional remarkable smooth vivid rich lifelike breathtaking",
            "negative": " blurry fuzzy pixelated grainy distorted unclear noisy low-resolution muddy dark washed-out dull smudged choppy patchy hazy unfocused overexposed underexposed faded low-quality subpar inferior mediocre flawed imperfect rough poor disappointing unacceptable"
        },
        "IQA2":
        {
            "positive": " stunning breathtaking mesmerizing dazzling sharp vivid ultra-clear pristine flawless cinematic professional crisp lifelike photorealistic vibrant rich detailed exquisite polished refined superb premium outstanding excellent magnificent superb impressive superior top-notch impeccable",
            "negative": " blurry distorted pixelated grainy fuzzy smeared muddy hazy unfocused low-res choppy jagged noisy patchy washed-out dull faded overexposed underexposed compressed artifacted glitchy low-grade subpar shoddy amateurish poor disappointing unacceptable terrible",

        }
    }
    
    if TASK not in text_dict:
        raise ValueError(f"Unknown task: {TASK}. Available tasks: {list(text_dict.keys())}")

    for name, words in text_dict[TASK].items():
        # Tokenize
        inputs = model.tokenizer(words, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]  # shape: (1, sequence_length)

        # 获取 Embedding
        embedding_layer = model.LLM.get_input_embeddings()
        embeddings[name] = embedding_layer(input_ids)

    # 计算平均 Embedding 并求差
    positive_vector = embeddings["positive"].mean(dim=1)  
    negative_vector = embeddings["negative"].mean(dim=1)  
    val_embed = positive_vector - negative_vector
    
    return val_embed


if __name__ == "__main__":
    # ================= 配置部分 =================
    TASK = "IQA"
    YML_FILE = {
        "IQA": "iqa.yml",
        "IAA": "iaa.yml",
    }
    MODEL_PATH = "iic/mPLUG-Owl3-7B-241101"
    
    # ================= 初始化 =================
    print(f"Initializing model for task: {TASK}...")
    
    # 加载数据集配置
    data_yml = YML_FILE[TASK]
    opt = yaml.safe_load(open(data_yml, "r"))
    val_datasets = {}
    for phase, datasets in opt.items():
        if phase not in ["train", "test"]:
            continue
        val_datasets[phase] = {}
        for name, data_args in datasets.items():
            # 动态加载数据集类
            dataset_cls = globals().get(data_args["type"])
            if dataset_cls:
                val_datasets[phase][name] = dataset_cls(**data_args["args"])
            else:
                print(f"Warning: Dataset class {data_args['type']} not found.")

    # 加载模型
    model = MultimodalQualityEvaluator(TASK, model_path=MODEL_PATH).to(device)
    model.eval()

    # 获取 Anchor Embedding
    val_embed = get_embed(model, device, TASK=TASK)
    print("Model and embeddings initialized.")

    # ================= 功能开关 =================
    # 根据需要取消注释以下代码块以运行特定功能

    # ---------------- 1. 多 Embedding & K 值评估 ----------------
    # val_embeds = []
    # for cat in ["IQA2", "IQA1"]:
    #     # ... (获取不同类别的 embedding)
    #     # val_embeds.append(val_embed)
    
    # all_embed_k_results = []
    # for name, val_dataset in val_datasets["test"].items():
    #     # ... (调用 evaluate_multiple_embeds)
    #     pass

    # ---------------- 2. Prompt 评估 ----------------
    # prompts_to_test = [
    #     {"name": "original", "user_content": "...", "assistant_content": "..."},
    #     # ...
    # ]
    # for name, val_dataset in val_datasets["test"].items():
    #     # ... (调用 prompt_evaluate)
    #     pass

    # ---------------- 3. Top-K Logits 分析 ----------------
    print("\nRunning Top-K Logits Analysis...")
    test_img_dir = "/home/ippl/xxr/mPLUG-Owl/mPLUG-Owl3/assets/testimg"
    if os.path.exists(test_img_dir):
        get_top_logits(test_img_dir, model, batch_size=1, topk=20)
    else:
        print(f"Test image directory not found: {test_img_dir}")

    # ---------------- 4. Q-Align & Q-Bench 评估 ----------------
    # print("\nRunning Q-Align & Q-Bench Evaluation...")
    # for name, val_dataset in val_datasets["test"].items():
    #     results, exp_data = q_evaluate(model, val_dataset, val_embed, bsz=1)
    #     print(f"******** {name} Results: *********")
    #     for key, value in results.items():
    #         print(f"{key}:\t {value:.4f}")
        
    #     # 保存实验数据
    #     os.makedirs("exps", exist_ok=True)
    #     exp_data_df = pd.DataFrame(exp_data)
    #     exp_data_df.to_csv(f"exps/exp_data_{name}.csv", index=False)
    #     print(f"Saved results to exps/exp_data_{name}.csv")

    # ---------------- 5. TopkDataset 评估 (基于预计算数据) ----------------
    # for name, val_dataset in val_datasets["test"].items():
    #     pre_save_path = f"exps/topk/{name}.pt"
    #     if os.path.exists(pre_save_path):
    #         # ... (加载数据并评估)
    #         pass
    
    print("\nDone.")