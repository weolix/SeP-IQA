import argparse
import os
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import tqdm
from scipy.stats import spearmanr, pearsonr
import pandas as pd

# Local imports
import data_loader
from data_loader import list_collate
import models_other

def get_embed(model, device, TASK="IQA"):
    """
    Get task-specific Anchor Embeddings.
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
        }
    }
    
    if TASK not in text_dict:
        raise ValueError(f"Unknown task: {TASK}")

    for name, words in text_dict[TASK].items():
        inputs = model.tokenizer(words, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]
        
        # Use the unified get_input_embeddings method
        embedding_layer = model.get_input_embeddings()
        embeddings[name] = embedding_layer(input_ids)

    positive_vector = embeddings["positive"].mean(dim=1)  
    negative_vector = embeddings["negative"].mean(dim=1)  
    val_embed = positive_vector - negative_vector
    
    return val_embed

def evaluate_model(model, val_dataset, val_embed, q_bench_tokens, q_align_tokens, bsz=2):
    model.eval()
    # num_workers = min(4, (os.cpu_count() // 2) if os.cpu_count() else 1)
    num_workers = 2 # Safe default
    valdataloader = DataLoader(val_dataset, batch_size=bsz, shuffle=False, num_workers=num_workers, collate_fn=list_collate)
    
    val_pred_scores_topk, qbench_val_scores, qalign_val_scores, val_gt_scores = [], [], [], []

    with torch.no_grad():
        for i, batch in enumerate(tqdm.tqdm(valdataloader, desc="Validation", ncols=100)):
            image_or_video_data_batch, labels = batch[0], batch[1]
            if not image_or_video_data_batch: continue

            # Forward pass
            outputs = model(image_or_video_batch=image_or_video_data_batch)
            logits = outputs.logits.to(torch.float32)
            last_token_logits = logits[:, -1, :] 
            
            # 1. Top-k Embeddings Method
            k = 100
            topk_results = torch.topk(last_token_logits, k, dim=-1)
            top_k_logits, top_k_indices = topk_results.values, topk_results.indices

            embedding_layer = model.get_input_embeddings()
            top_k_embeddings = embedding_layer(top_k_indices.to(embedding_layer.weight.device))
            
            val_embed_unsqueezed = val_embed.to(top_k_embeddings.device).unsqueeze(0).unsqueeze(0)
            weights = F.cosine_similarity(top_k_embeddings, val_embed_unsqueezed, dim=-1)
            
            score_topk = torch.sum(top_k_logits.to(weights.device) * weights, dim=-1)
            val_pred_scores_topk.extend(score_topk.cpu().tolist())
            
            # 2. Q-Bench Method (good vs poor)
            if q_bench_tokens:
                q_bench_token_ids = torch.tensor(q_bench_tokens, device=last_token_logits.device).long()
                binary_logits = last_token_logits[:, q_bench_token_ids]
                binary_probability = torch.softmax(binary_logits, dim=-1) 
                q_bench_score = binary_probability[:, 0]  # probability of "good"
                qbench_val_scores.extend(q_bench_score.cpu().tolist())

            # 3. Q-Align Method (5-level rating)
            if q_align_tokens:
                q_align_token_ids = torch.tensor(q_align_tokens, device=last_token_logits.device).long()
                align_logits = last_token_logits[:, q_align_token_ids]
                align_probability = torch.softmax(align_logits, dim=-1)
                
                rating_weights = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device=align_probability.device)
                q_align_score = torch.sum(align_probability * rating_weights, dim=-1)
                qalign_val_scores.extend(q_align_score.cpu().tolist())

            val_gt_scores.extend(labels.cpu().tolist())
        
    # Calculate Metrics
    results_dict = {}
    val_gt_scores_tensor = torch.tensor(val_gt_scores, dtype=torch.float32)

    def calc_metrics(pred_scores, name):
        if not pred_scores:
            results_dict[f"srcc_{name}"] = 0.0
            results_dict[f"plcc_{name}"] = 0.0
            return

        pred_tensor = torch.tensor(pred_scores, dtype=torch.float32)
        valid_indices = ~torch.isnan(pred_tensor) & ~torch.isinf(pred_tensor)
        
        if not torch.all(valid_indices):
            print(f"Warning ({name}): Found NaN/Inf scores. Excluding them.")
            pred_tensor = pred_tensor[valid_indices]
            gt_tensor = val_gt_scores_tensor[valid_indices]
        else:
            gt_tensor = val_gt_scores_tensor

        if len(pred_tensor) >= 2:
            srcc, _ = spearmanr(pred_tensor.numpy(), gt_tensor.numpy())
            plcc, _ = pearsonr(pred_tensor.numpy(), gt_tensor.numpy())
            results_dict[f"srcc_{name}"] = srcc
            results_dict[f"plcc_{name}"] = plcc
        else:
            results_dict[f"srcc_{name}"] = 0.0
            results_dict[f"plcc_{name}"] = 0.0

    calc_metrics(val_pred_scores_topk, "topk")
    calc_metrics(qbench_val_scores, "qbench")
    calc_metrics(qalign_val_scores, "qalign")
        
    return results_dict

def load_datasets(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    opt = yaml.safe_load(open(config_path, "r"))
    datasets = {}
    for phase, phase_datasets in opt.items():
        if phase not in ["train", "test"]: continue
        datasets[phase] = {}
        for name, data_args in phase_datasets.items():
            dataset_cls = getattr(data_loader, data_args["type"], None)
            if dataset_cls:
                datasets[phase][name] = dataset_cls(**data_args["args"])
            else:
                print(f"Warning: Dataset class {data_args['type']} not found.")
    return datasets

def main():
    parser = argparse.ArgumentParser(description="Evaluate Other Base Models")
    parser.add_argument("--model_type", type=str, required=True, 
                        choices=["internvl3", "qwen2vl", "qwen25vl", "llava_video", "llava_next"],
                        help="Type of model to evaluate")
    parser.add_argument("--model_path", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--task", type=str, default="IQA", choices=["IQA", "IAA"], help="Task name")
    parser.add_argument("--config", type=str, default=None, help="Path to dataset config YAML")
    parser.add_argument("--output_dir", type=str, default="exps_other", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    
    args = parser.parse_args()
    
    # Default model paths
    default_paths = {
        "internvl3": "OpenGVLab/InternVL3-8B-hf",
        "qwen2vl": "Qwen/Qwen2-VL-7B-Instruct",
        "qwen25vl": "LLMs/Qwen2.5-VL-7B-Instruct",
        "llava_video": "llava-hf/llava-v1.6-mistral-7b-hf",
        "llava_next": "llava-hf/llava-v1.6-vicuna-7b-hf"
    }
    
    if args.model_path is None:
        args.model_path = default_paths.get(args.model_type)
        
    if args.config is None:
        args.config = "iqa.yml" if args.task == "IQA" else "iaa.yml"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize Model
    print(f"Initializing {args.model_type} from {args.model_path}...")
    if args.model_type == "internvl3":
        model = models_other.InternVL3Evaluator(args.task, args.model_path).to(device)
    elif args.model_type == "qwen2vl":
        model = models_other.Qwen2VLEvaluator(args.task, args.model_path).to(device)
    elif args.model_type == "qwen25vl":
        model = models_other.Qwen25VLEvaluator(args.task, args.model_path).to(device)
    elif args.model_type == "llava_video":
        model = models_other.LlavaVideoEvaluator(args.task, args.model_path).to(device)
    elif args.model_type == "llava_next":
        model = models_other.LlavaNextEvaluator(args.task, args.model_path).to(device)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    model.eval()
    
    # Prepare Tokens for Q-Bench and Q-Align
    tokenizer = model.tokenizer
    try:
        q_bench_tokens = [tokenizer.encode(t, add_special_tokens=False)[0] for t in ["good", "poor"]]
        q_align_tokens = [tokenizer.encode(t, add_special_tokens=False)[0] for t in ["bad", "poor", "fair", "good", "excellent"]]
    except Exception as e:
        print(f"Warning: Could not encode Q-Bench/Q-Align tokens: {e}")
        q_bench_tokens = None
        q_align_tokens = None

    # Get Anchor Embedding
    try:
        val_embed = get_embed(model, device, TASK=args.task)
    except Exception as e:
        print(f"Warning: Could not get anchor embeddings: {e}")
        val_embed = torch.zeros(1, device=device) # Dummy

    # Load Datasets
    print(f"Loading datasets from {args.config}...")
    datasets = load_datasets(args.config)
    if "test" not in datasets:
        print("No test datasets found.")
        return
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run Evaluation
    for name, val_dataset in datasets["test"].items():
        print(f"Evaluating dataset: {name}")
        results = evaluate_model(model, val_dataset, val_embed, q_bench_tokens, q_align_tokens, bsz=args.batch_size)
        
        print(f"******** {name} Results: *********")
        for key, value in results.items():
            print(f"{key}:\t {value:.4f}")
            
        # Save results
        save_path = os.path.join(args.output_dir, f"results_{args.model_type}_{name}.json")
        import json
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=4)
            
    print("\nDone.")

if __name__ == "__main__":
    main()
