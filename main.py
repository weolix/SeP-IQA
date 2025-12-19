import argparse
import os
import yaml
import torch
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import time

# Local imports
from models import MultimodalQualityEvaluator
import data_loader
from evaluation_utils import get_embed
import evaluators

def load_datasets(config_path):
    """Load datasets from a YAML configuration file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
        
    opt = yaml.safe_load(open(config_path, "r"))
    datasets = {}
    
    for phase, phase_datasets in opt.items():
        if phase not in ["train", "test"]:
            continue
        datasets[phase] = {}
        for name, data_args in phase_datasets.items():
            # Dynamically load dataset class from data_loader module
            dataset_cls = getattr(data_loader, data_args["type"], None)
            if dataset_cls:
                datasets[phase][name] = dataset_cls(**data_args["args"])
            else:
                print(f"Warning: Dataset class {data_args['type']} not found in data_loader.")
                
    return datasets

def main():
    parser = argparse.ArgumentParser(description="mPLUG-Owl3 Zero-shot Quality Assessment")
    
    # General arguments
    parser.add_argument("--task", type=str, default="IQA", choices=["IQA", "IAA"], help="Task name")
    parser.add_argument("--config", type=str, default=None, help="Path to dataset config YAML file")
    parser.add_argument("--model_path", type=str, default="iic/mPLUG-Owl3-7B-241101", help="Path to model")
    parser.add_argument("--mode", type=str, required=True, 
                        choices=["prompt", "embed", "q_align", "topk_logits", "fit"], 
                        help="Evaluation mode")
    parser.add_argument("--output_dir", type=str, default="exps", help="Directory to save results")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    
    # Mode-specific arguments
    parser.add_argument("--img_dir", type=str, default=None, help="Image directory for topk_logits mode")
    parser.add_argument("--topk", type=int, default=20, help="Top-K for topk_logits mode")
    parser.add_argument("--prompts_file", type=str, default=None, help="JSON file containing prompts for prompt mode")
    
    args = parser.parse_args()
    
    # Set up device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Determine config file
    if args.config is None:
        args.config = "iqa.yml" if args.task == "IQA" else "iaa.yml"
        
    # Initialize model
    print(f"Initializing model from {args.model_path}...")
    model = MultimodalQualityEvaluator(args.task, model_path=args.model_path).to(device)
    model.eval()
    
    # Get Anchor Embedding
    val_embed = get_embed(model, device, TASK=args.task)
    print("Model and embeddings initialized.")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Execute based on mode
    if args.mode == "topk_logits":
        if args.img_dir is None:
            # Default to testimg if not specified
            args.img_dir = os.path.join(os.path.dirname(args.model_path), "../mPLUG-Owl3/assets/testimg")
            if not os.path.exists(args.img_dir):
                 # Fallback to current directory or raise error
                 print("Please specify --img_dir")
                 return

        print(f"Running Top-K Logits Analysis on {args.img_dir}...")
        evaluators.get_top_logits(args.img_dir, model, batch_size=args.batch_size, topk=args.topk)
        
    else:
        # Load datasets for other modes
        print(f"Loading datasets from {args.config}...")
        datasets = load_datasets(args.config)
        if "test" not in datasets:
            print("No test datasets found in config.")
            return
            
        test_datasets = datasets["test"]
        
        if args.mode == "q_align":
            print("\nRunning Q-Align & Q-Bench Evaluation...")
            for name, val_dataset in test_datasets.items():
                print(f"Evaluating dataset: {name}")
                results, exp_data = evaluators.q_evaluate(model, val_dataset, val_embed, bsz=args.batch_size)
                
                print(f"******** {name} Results: *********")
                for key, value in results.items():
                    print(f"{key}:\t {value:.4f}")
                
                # Save results
                save_path = os.path.join(args.output_dir, f"exp_data_{name}.csv")
                pd.DataFrame(exp_data).to_csv(save_path, index=False)
                print(f"Saved results to {save_path}")
                
        elif args.mode == "embed":
            print("\nRunning Embedding Evaluation...")
            # Note: This currently runs a simplified version. 
            # For full multiple embedding evaluation, we might need more arguments or a specific config.
            # Here we use the default val_embed.
            
            # If you want to test multiple embeddings as in the original script, 
            # you would need to define them here or load them.
            # For now, let's run evaluate_multiple_embeds with the single default embed as a list
            
            val_embeds = [val_embed]
            
            for name, val_dataset in test_datasets.items():
                print(f"Evaluating dataset: {name}")
                results = evaluators.evaluate_multiple_embeds(model, val_dataset, val_embeds, bsz=args.batch_size)
                
                # Print results
                for embed_name, res in results.items():
                    print(f"  {embed_name}:")
                    for k, metrics in res.items():
                        print(f"    K={k}: SRCC={metrics['srcc']:.4f}, PLCC={metrics['plcc']:.4f}")
                        
        elif args.mode == "prompt":
            print("\nRunning Prompt Evaluation...")
            if args.prompts_file:
                import json
                with open(args.prompts_file, 'r') as f:
                    prompts_list = json.load(f)
            else:
                # Default prompt
                prompts_list = [{
                    "name": "default",
                    "user_content": "Rate the quality of the image.",
                    "assistant_content": "The quality of the image is"
                }]
                print("Using default prompt. Provide --prompts_file for more.")
                
            for name, val_dataset in test_datasets.items():
                print(f"Evaluating dataset: {name}")
                results = evaluators.prompt_evaluate(model, val_dataset, val_embed, prompts_list, bsz=args.batch_size)
                
                # Save results
                save_path = os.path.join(args.output_dir, f"prompt_eval_{name}.json")
                # Convert tensor values to float for JSON serialization
                # (This is handled in prompt_evaluate but let's be safe)
                import json
                class NumpyEncoder(json.JSONEncoder):
                    def default(self, obj):
                        if isinstance(obj, (torch.Tensor, np.ndarray)):
                            return obj.tolist()
                        return super().default(obj)

                with open(save_path, 'w') as f:
                    json.dump(results, f, cls=NumpyEncoder, indent=4)
                print(f"Saved results to {save_path}")

        elif args.mode == "fit":
             print("\nRunning Embedding Fitting...")
             for name, val_dataset in test_datasets.items():
                print(f"Fitting on dataset: {name}")
                # Note: embed_fit returns (val_embed, topk_data)
                # It modifies the model or embedding. 
                # This is experimental as per original script.
                new_embed, topk_data = evaluators.embed_fit(model, val_dataset, val_embed, bsz=args.batch_size)
                
                if topk_data:
                    save_path = os.path.join(args.output_dir, f"fitted_data_{name}.pt")
                    torch.save(topk_data, save_path)
                    print(f"Saved fitted data to {save_path}")

    print("\nDone.")

if __name__ == "__main__":
    main()
