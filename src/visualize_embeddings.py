import torch
import transformers
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import os
import argparse
import sys

# Add current directory to path to allow imports if needed
sys.path.append(os.getcwd())

# Import the model class. 
# Assuming owl3_zeroshot.py is in the same directory and can be imported.
try:
    from owl3_zeroshot import MultimodalQualityEvaluator
except ImportError:
    # If import fails (e.g. due to missing dependencies in environment for other parts of the file),
    # we might need to define the class here or mock it. 
    # But let's assume it works as it is in the workspace.
    print("Could not import MultimodalQualityEvaluator from owl3_zeroshot.py. Please ensure the file exists and dependencies are met.")
    sys.exit(1)

# Define the text dictionary as in owl3_zeroshot.py
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

def get_embeddings_and_centroid(model, words_str, device):
    """
    Get embeddings for individual words and the centroid of the whole string.
    """
    # 1. Calculate the TRUE centroid (as used in the model)
    # The model tokenizes the whole string and takes the mean of all tokens.
    inputs = model.tokenizer(words_str, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"] # shape: (1, seq_len)
    
    embedding_layer = model.LLM.get_input_embeddings()
    all_token_embeds = embedding_layer(input_ids)[0] # shape: (seq_len, hidden_dim)
    
    true_centroid = all_token_embeds.mean(dim=0).float().detach().cpu().numpy()
    
    # 2. Get embeddings for individual words
    word_list = words_str.strip().split()
    word_embeddings = []
    valid_words = []
    
    for word in word_list:
        # We prepend a space to match the likely tokenization in the sentence
        # (assuming the words in the string were separated by spaces)
        word_input = " " + word
        
        # Tokenize the single word
        w_inputs = model.tokenizer(word_input, return_tensors="pt").to(device)
        w_ids = w_inputs["input_ids"]
        
        # Get embedding
        w_embeds = embedding_layer(w_ids)[0] # shape: (w_seq_len, hidden_dim)
        
        # Average tokens for this word to get a single vector
        w_mean = w_embeds.mean(dim=0).float().detach().cpu().numpy()
        
        word_embeddings.append(w_mean)
        valid_words.append(word)
        
    return np.array(word_embeddings), valid_words, true_centroid

def visualize_embeddings(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading model from {args.model_path}...")
    # Initialize model
    try:
        model = MultimodalQualityEvaluator(task=args.task, model_path=args.model_path).to(device)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    task_words = text_dict.get(args.task)
    if not task_words:
        print(f"Task {args.task} not found in text_dict.")
        return

    print(f"Extracting embeddings for task: {args.task}")
    
    pos_words_str = task_words["positive"]
    neg_words_str = task_words["negative"]

    pos_embeddings, pos_word_list, pos_centroid = get_embeddings_and_centroid(model, pos_words_str, device)
    neg_embeddings, neg_word_list, neg_centroid = get_embeddings_and_centroid(model, neg_words_str, device)

    # Combine all embeddings for dimensionality reduction
    # Structure: [Pos Words, Neg Words, Pos Centroid, Neg Centroid]
    all_embeddings = np.vstack([
        pos_embeddings, 
        neg_embeddings, 
        pos_centroid.reshape(1, -1), 
        neg_centroid.reshape(1, -1)
    ])
    
    print(f"Total embeddings: {len(all_embeddings)}")
    print(f"Reducing dimensionality using {args.method}...")
    
    n_components = 3 if args.dim == 3 else 2

    if args.method == 'tsne':
        # Perplexity must be less than number of samples
        n_samples = len(all_embeddings)
        if args.perplexity is None:
            perplexity = min(30, n_samples - 1) if n_samples > 1 else 1
        else:
            perplexity = args.perplexity
            
        # Use cosine metric for better semantic separation
        metric = args.metric
        init = 'pca' if metric == 'euclidean' else 'random' # PCA init is typically for euclidean
        
        reducer = TSNE(n_components=n_components, random_state=42, perplexity=perplexity, 
                      metric=metric, init=init, learning_rate='auto')
        reduced_embeddings = reducer.fit_transform(all_embeddings)
        
    elif args.method == 'pca':
        reducer = PCA(n_components=n_components)
        reduced_embeddings = reducer.fit_transform(all_embeddings)
        
    elif args.method == 'quality_axis':
        # Custom projection: X-axis is the vector connecting centroids (Quality Axis)
        # Y and Z axes are PCA of the residuals
        
        # 1. Calculate difference vector (Quality Axis)
        diff_vector = pos_centroid - neg_centroid # shape (dim,)
        diff_vector = diff_vector / np.linalg.norm(diff_vector)
        
        # 2. Project all embeddings onto diff_vector (X-axis)
        x_coords = np.dot(all_embeddings, diff_vector)
        
        # 3. Calculate residuals (remove component along Quality Axis)
        projection = np.outer(x_coords, diff_vector)
        residuals = all_embeddings - projection
        
        # 4. PCA on residuals for remaining dimensions
        pca = PCA(n_components=n_components - 1)
        other_coords = pca.fit_transform(residuals)
        
        # Combine
        reduced_embeddings = np.hstack([x_coords.reshape(-1, 1), other_coords])
        
        print("Using Quality Axis projection: X-axis is alignment with (Pos - Neg) centroid vector.")

    # Split back
    n_pos = len(pos_embeddings)
    n_neg = len(neg_embeddings)
    
    pos_reduced = reduced_embeddings[:n_pos]
    neg_reduced = reduced_embeddings[n_pos:n_pos+n_neg]
    pos_centroid_reduced = reduced_embeddings[-2]
    neg_centroid_reduced = reduced_embeddings[-1]

    if args.dim == 3:
        # 3D Plotting using Plotly
        fig = go.Figure()

        # Positive Words
        fig.add_trace(go.Scatter3d(
            x=pos_reduced[:, 0],
            y=pos_reduced[:, 1],
            z=pos_reduced[:, 2],
            mode='markers+text',
            name='Positive Words',
            text=pos_word_list,
            textposition="top center",
            marker=dict(
                size=5,
                color='blue',
                opacity=0.6
            )
        ))

        # Negative Words
        fig.add_trace(go.Scatter3d(
            x=neg_reduced[:, 0],
            y=neg_reduced[:, 1],
            z=neg_reduced[:, 2],
            mode='markers+text',
            name='Negative Words',
            text=neg_word_list,
            textposition="top center",
            marker=dict(
                size=5,
                color='red',
                opacity=0.6
            )
        ))

        # Positive Centroid
        fig.add_trace(go.Scatter3d(
            x=[pos_centroid_reduced[0]],
            y=[pos_centroid_reduced[1]],
            z=[pos_centroid_reduced[2]],
            mode='markers',
            name='Positive Centroid',
            marker=dict(
                size=15,
                color='gold',
                symbol='diamond',
                line=dict(
                    color='black',
                    width=2
                )
            )
        ))

        # Negative Centroid
        fig.add_trace(go.Scatter3d(
            x=[neg_centroid_reduced[0]],
            y=[neg_centroid_reduced[1]],
            z=[neg_centroid_reduced[2]],
            mode='markers',
            name='Negative Centroid',
            marker=dict(
                size=15,
                color='purple',
                symbol='diamond',
                line=dict(
                    color='black',
                    width=2
                )
            )
        ))

        fig.update_layout(
            title=f'3D Embedding Distribution Visualization ({args.method.upper()}) - {args.task}',
            scene=dict(
                xaxis_title='Component 1',
                yaxis_title='Component 2',
                zaxis_title='Component 3'
            ),
            margin=dict(l=0, r=0, b=0, t=40)
        )

        output_file = args.output
        if not output_file:
            output_file = f'embedding_distribution_{args.task}_{args.method}_3d.html'
        
        fig.write_html(output_file)
        print(f"3D Plot saved to {output_file}")

    else:
        # 2D Plotting using Matplotlib
        plt.figure(figsize=(14, 10))
        
        # Plot positive words
        plt.scatter(pos_reduced[:, 0], pos_reduced[:, 1], c='blue', label='Positive Words', alpha=0.5, s=80, edgecolors='none')
        for i, word in enumerate(pos_word_list):
            plt.annotate(word, (pos_reduced[i, 0], pos_reduced[i, 1]), fontsize=9, alpha=0.8, color='darkblue')

        # Plot negative words
        plt.scatter(neg_reduced[:, 0], neg_reduced[:, 1], c='red', label='Negative Words', alpha=0.5, s=80, edgecolors='none')
        for i, word in enumerate(neg_word_list):
            plt.annotate(word, (neg_reduced[i, 0], neg_reduced[i, 1]), fontsize=9, alpha=0.8, color='darkred')

        # Plot centroids
        plt.scatter(pos_centroid_reduced[0], pos_centroid_reduced[1], c='gold', marker='*', s=400, edgecolors='black', linewidth=1.5, label='Positive Centroid')
        plt.scatter(neg_centroid_reduced[0], neg_centroid_reduced[1], c='purple', marker='*', s=400, edgecolors='black', linewidth=1.5, label='Negative Centroid')

        plt.title(f'Embedding Distribution Visualization ({args.method.upper()}) - {args.task}', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        
        output_file = args.output
        if not output_file:
            output_file = f'embedding_distribution_{args.task}_{args.method}.png'
            
        plt.savefig(output_file, dpi=300)
        print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize embedding distribution")
    parser.add_argument("--model-path", type=str, default="iic/mPLUG-Owl3-7B-241101", help="Path to the model")
    parser.add_argument("--task", type=str, default="IQA", help="Task name (IQA, IAA, etc.)")
    parser.add_argument("--method", type=str, default="tsne", choices=["tsne", "pca", "quality_axis"], help="Dimensionality reduction method")
    parser.add_argument("--dim", type=int, default=2, choices=[2, 3], help="Dimension of the visualization (2 or 3)")
    parser.add_argument("--perplexity", type=float, default=None, help="Perplexity for t-SNE")
    parser.add_argument("--metric", type=str, default="cosine", choices=["euclidean", "cosine"], help="Metric for t-SNE")
    parser.add_argument("--output", type=str, default="", help="Output image file path")
    
    args = parser.parse_args()
    visualize_embeddings(args)
