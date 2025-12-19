import torch
import json
import numpy as np

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

class NumpyEncoder(json.JSONEncoder):
    """
    Numpy JSON 编码器。
    用于解决 json.dump 无法直接序列化 numpy 数据类型的问题。
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

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
