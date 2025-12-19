import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import tqdm
from scipy.stats import spearmanr, pearsonr
from PIL import Image
import glob
import gc

# Local imports
from data_loader import TopkDataset, list_collate
from evaluation_utils import plcc_loss

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
        valdataloader = DataLoader(val_dataset, batch_size=bsz, shuffle=False, num_workers=4, collate_fn=list_collate)
    
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
        top_k_embeddings = embedding_layer(top_k_indices.to(model.dev))
        val_embed_unsqueezed = val_embed.to(top_k_embeddings.device).unsqueeze(0).unsqueeze(0)

        weights = F.cosine_similarity(top_k_embeddings, val_embed_unsqueezed, dim=-1)
        weighted_logits = top_k_logits.to(model.dev) * weights # Shape: (batch_size, k)
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
    """
    
    model.eval()
    
    # 如果没有指定 collect_bsz，使用更小的 batch size 进行数据收集以节省显存
    if collect_bsz is None:
        collect_bsz = max(1, bsz // 2)

    collect_dataloader = DataLoader(val_dataset, batch_size=collect_bsz, shuffle=False, num_workers=2, collate_fn=val_dataset.collate_fn)

    all_prompt_results = {}
    
    # 1. 预先收集真实标签 (Ground Truth) 和数据索引
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
    """
    model.eval()
    valdataloader = DataLoader(val_dataset, batch_size=bsz, shuffle=False, num_workers=4, collate_fn=list_collate)
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
                    
                    batch_logits = all_logits[start_idx:end_idx].to(model.dev)
                    
                    # 获取 Top-K Logits 和 Indices
                    topk_output = torch.topk(batch_logits, K, dim=-1)
                    top_k_logits = topk_output.values    # Shape: (batch_size, k)
                    top_k_indices = topk_output.indices  # Shape: (batch_size, k)

                    # 获取词嵌入
                    embedding_layer = model.LLM.get_input_embeddings()
                    top_k_embeddings = embedding_layer(top_k_indices)  # Shape: (batch_size, k, embedding_dim)

                    # 计算余弦相似度权重
                    val_embed_unsqueezed = val_embed.to(model.dev).unsqueeze(0).unsqueeze(0)
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

def q_evaluate(model, val_dataset, val_embed, bsz=2):
    """
    综合评估函数。
    同时评估 Embedding 方法 (New Method), Q-Bench 方法, 和 Q-Align 方法。
    """
    model.eval()
    
    # Q-Bench 和 Q-Align 的特定 Token ID
    qbench_tokens = [model.tokenizer.encode(t)[0] for t in ["good", "poor"]]
    qalign_tokens = [model.tokenizer.encode(t)[0] for t in ["bad", "poor", "fair", "good", "excellent"]]
    
    valdataloader = DataLoader(val_dataset, batch_size=bsz, shuffle=False, num_workers=0, collate_fn=list_collate)
    
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
    """
    
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
