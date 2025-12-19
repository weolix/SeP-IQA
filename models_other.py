import torch
import torch.nn as nn
import transformers
from PIL import Image
from qwen_vl_utils import process_vision_info
import os

# =================================================================================================
# InternVL3 Evaluator
# =================================================================================================
class InternVL3Evaluator(nn.Module):
    def __init__(self, task="IQA", model_path="OpenGVLab/InternVL3-8B-hf", local_files_only=False): 
        super().__init__()
        self.task = task
        
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        if not torch.cuda.is_available():
            dtype = torch.float32
        
        print(f"InternVL3 model will be loaded with dtype: {dtype}")

        self.model = transformers.AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
        
        self.processor = transformers.AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
        
        self.tokenizer = self.processor.tokenizer

    def get_input_embeddings(self):
        if hasattr(self.model, 'language_model'):
            return self.model.language_model.get_input_embeddings()
        return self.model.get_input_embeddings()

    def _prepare_messages(self, data_item):
        media_data = data_item["data"]
        is_video = isinstance(media_data, list) and all(isinstance(frame, Image.Image) for frame in media_data)
        media_type_str = "video" if is_video else "image"

        if is_video:
            content_item = {
                "type": "image", 
                "url": media_data[0] if len(media_data) > 0 else None, 
            }
        else:
            content_item = {
                "type": "image",
                "url": media_data,
            }

        if self.task == "IQA":
            user_text = f"How would you rate the quality of this {media_type_str}?"
            assistant_prefix = f"The quality of this {media_type_str} is very"
            messages = [{"role": "user", "content": [content_item, {"type": "text", "text": user_text}]}]
        elif self.task == "IAA":
            user_text = f"Considering its artistic composition, color harmony, and overall visual appeal, use an adjective to describe the aesthetic quality of this {media_type_str}?"
            assistant_prefix = f"The {media_type_str} is"
            messages = [{"role": "user", "content": [content_item, {"type": "text", "text": user_text}]}]
        else:
            raise ValueError(f"Unknown task: {self.task}")
        
        return messages, assistant_prefix

    def forward(self, image_or_video_batch=None, labels=None, **args):
        batch_messages = []
        batch_assistant_prefixes = []

        for item in image_or_video_batch:
            messages, assistant_prefix = self._prepare_messages(item)
            batch_messages.append(messages)
            batch_assistant_prefixes.append(assistant_prefix)

        inputs = self.processor.apply_chat_template(
            batch_messages,
            padding=True, 
            add_generation_prompt=True, 
            tokenize=True, 
            return_dict=True, 
            return_tensors="pt"
        )
        
        batch_size = len(batch_assistant_prefixes)
        new_input_ids = []
        new_attention_masks = []
        
        for i in range(batch_size):
            current_input_ids = inputs['input_ids'][i]
            current_attention_mask = inputs['attention_mask'][i] if 'attention_mask' in inputs else None
            
            if current_attention_mask is not None:
                valid_length = current_attention_mask.sum().item()
                actual_input_ids = current_input_ids[:valid_length]
            else:
                actual_input_ids = current_input_ids
            
            prefix_tokens = self.processor.tokenizer.encode(batch_assistant_prefixes[i], add_special_tokens=False)
            prefix_tensor = torch.tensor(prefix_tokens, dtype=torch.long)
            
            new_input_ids_sample = torch.cat([actual_input_ids, prefix_tensor.to(current_input_ids.device)])
            new_input_ids.append(new_input_ids_sample)
            new_attention_masks.append(torch.ones(len(new_input_ids_sample), dtype=torch.long))
        
        max_new_length = max(len(ids) for ids in new_input_ids)
        pad_token_id = self.processor.tokenizer.pad_token_id or self.processor.tokenizer.eos_token_id or 0
        
        padded_input_ids = []
        padded_attention_masks = []
        
        for i in range(batch_size):
            current_length = len(new_input_ids[i])
            padding_length = max_new_length - current_length
            
            if padding_length > 0:
                padding_ids = torch.full((padding_length,), pad_token_id, dtype=torch.long, device=new_input_ids[i].device)
                padding_mask = torch.zeros(padding_length, dtype=torch.long, device=new_input_ids[i].device)
                padded_input_ids.append(torch.cat([padding_ids, new_input_ids[i]]))
                padded_attention_masks.append(torch.cat([padding_mask, new_attention_masks[i].to(new_input_ids[i].device)]))
            else:
                padded_input_ids.append(new_input_ids[i])
                padded_attention_masks.append(new_attention_masks[i].to(new_input_ids[i].device))
                
        final_input_ids = torch.stack(padded_input_ids)
        final_attention_mask = torch.stack(padded_attention_masks)
        
        pixel_values = inputs['pixel_values']
        image_grid_thw = inputs.get('image_grid_thw', None)
        
        model_inputs = {
            "input_ids": final_input_ids.to(self.model.device),
            "attention_mask": final_attention_mask.to(self.model.device),
            "pixel_values": pixel_values.to(self.model.device),
        }
        if image_grid_thw is not None:
            model_inputs["image_grid_thw"] = image_grid_thw.to(self.model.device)

        return self.model(**model_inputs)

# =================================================================================================
# Qwen2VL Evaluator
# =================================================================================================
class Qwen2VLEvaluator(nn.Module):
    def __init__(self, task="IQA", model_path="Qwen/Qwen2-VL-7B-Instruct", local_files_only=False):
        super().__init__()
        self.task = task
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        if not torch.cuda.is_available(): dtype = torch.float32
        
        print(f"Qwen2-VL model will be loaded with dtype: {dtype}")
        self.LLM = transformers.Qwen2VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=dtype, device_map="auto", trust_remote_code=True, local_files_only=local_files_only
        )
        self.LLMprocessor = transformers.AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True, local_files_only=local_files_only
        )
        self.tokenizer = self.LLMprocessor.tokenizer

    def get_input_embeddings(self):
        return self.LLM.get_input_embeddings()

    def _prepare_messages(self, data_item):
        media_data = data_item["data"]
        is_video = isinstance(media_data, list) and all(isinstance(frame, Image.Image) for frame in media_data)
        media_type_str = "video" if is_video else "image"

        if is_video:
            content_item = {"type": "video", "video": media_data, "fps": 1.0}
        else:
            content_item = {"type": "image", "image": media_data}

        if self.task == "IQA":
            user_text = f"Taking into account the details and the rationality of the {media_type_str}, how would you rate the quality of this {media_type_str}?"
            assistant_prefix = f"The quality of this {media_type_str} is"
            messages = [{"role": "user", "content": [content_item, {"type": "text", "text": user_text}]}]
        elif self.task == "IAA":
            user_text = f"Considering its artistic composition, color harmony, and overall visual appeal, use an adjective to describe the aesthetic quality of this {media_type_str}?"
            assistant_prefix = f"The {media_type_str} is"
            system_prompt = "You are a demanding art critic..."
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": [content_item, {"type": "text", "text": user_text}]}]
        else:
            raise ValueError(f"Unknown task: {self.task}")
        return messages, assistant_prefix

    def forward(self, image_or_video_batch=None, labels=None, **args):
        batch_messages = []
        batch_assistant_prefixes = []
        for item in image_or_video_batch:
            messages, assistant_prefix = self._prepare_messages(item)
            batch_messages.append(messages)
            batch_assistant_prefixes.append(assistant_prefix)

        batch_texts = []
        for i, messages in enumerate(batch_messages):
            text = self.LLMprocessor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            text = text + batch_assistant_prefixes[i]
            batch_texts.append(text)

        image_inputs, video_inputs = process_vision_info(batch_messages)
        inputs = self.LLMprocessor(text=batch_texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
        
        device = self.LLM.device
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor): inputs[k] = v.to(device)
            
        return self.LLM(**inputs)

# =================================================================================================
# Qwen2.5VL Evaluator
# =================================================================================================
class Qwen25VLEvaluator(nn.Module):
    def __init__(self, task="IQA", model_path="LLMs/Qwen2.5-VL-7B-Instruct", local_files_only=True):
        super().__init__()
        self.task = task
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        if not torch.cuda.is_available(): dtype = torch.float32
        
        print(f"Qwen2.5-VL model will be loaded with dtype: {dtype}")
        self.LLM = transformers.Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, trust_remote_code=True, local_files_only=local_files_only,
            attn_implementation="sdpa", torch_dtype=dtype, device_map="auto"
        )
        self.LLMprocessor = transformers.AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True, local_files_only=local_files_only, use_fast=True
        )
        self.tokenizer = self.LLMprocessor.tokenizer

    def get_input_embeddings(self):
        return self.LLM.get_input_embeddings()

    def _prepare_messages(self, data_item):
        media_data = data_item["data"]
        is_video = isinstance(media_data, list) and all(isinstance(frame, Image.Image) for frame in media_data)
        media_type_key = "video" if is_video else "image"
        media_type_str = "video" if is_video else "image"

        if self.task == "IQA":
            user_text = f"Taking into account the details and the rationality of the {media_type_str}, how would you rate the quality of this {media_type_str}?"
            assistant_prefix = f"The quality of this {media_type_str} is"
            messages = [{"role": "user", "content": [{"type": media_type_key, media_type_key: media_data}, {"type": "text", "text": user_text}]}]
        elif self.task == "IAA":
            user_text = f"Considering its artistic composition, color harmony, and overall visual appeal, use an adjective to describe the aesthetic quality of this {media_type_str}?"
            assistant_prefix = f"The {media_type_str} is"
            system_prompt = "You are a demanding art critic..."
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": [{"type": media_type_key, media_type_key: media_data}, {"type": "text", "text": user_text}]}]
        else:
            raise ValueError(f"Unknown task: {self.task}")
        return messages, assistant_prefix

    def forward(self, image_or_video_batch=None, labels=None, **args):
        batch_messages = []
        batch_assistant_prefixes = []
        for item in image_or_video_batch:
            messages, assistant_prefix = self._prepare_messages(item)
            batch_messages.append(messages)
            batch_assistant_prefixes.append(assistant_prefix)

        batch_texts = []
        for i, messages in enumerate(batch_messages):
            text = self.LLMprocessor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            text = text + batch_assistant_prefixes[i]
            batch_texts.append(text)

        image_inputs, video_inputs = process_vision_info(batch_messages)
        inputs = self.LLMprocessor(text=batch_texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
        
        device = self.LLM.device
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor): inputs[k] = v.to(device)
            
        return self.LLM(**inputs)

# =================================================================================================
# LLaVA Video Evaluator
# =================================================================================================
class LlavaVideoEvaluator(nn.Module):
    def __init__(self, task="IQA", model_path="llava-hf/llava-v1.6-mistral-7b-hf", local_files_only=False):
        super().__init__()
        self.task = task
        print(f"Loading LLaVA Video model from {model_path}")
        
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        if not torch.cuda.is_available(): dtype = torch.float32
        
        self.model = transformers.LlavaNextVideoForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=dtype, device_map="auto", local_files_only=local_files_only,
            attn_implementation="sdpa"
        )
        self.processor = transformers.LlavaNextVideoProcessor.from_pretrained(
            model_path, local_files_only=local_files_only, patch_size=14, use_fast=True
        )
        self.tokenizer = self.processor.tokenizer

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def _prepare_prompt_and_media(self, data_item):
        media_data = data_item["data"]
        is_video = isinstance(media_data, list) and all(isinstance(frame, Image.Image) for frame in media_data)
        media_type_placeholder = "video" if is_video else "image"
        media_type_str = "video" if is_video else "image"

        if self.task == "IQA":
            user_prompt_text = f"Taking into account the details and the rationality of the {media_type_str}, how would you rate the quality of this {media_type_str}?"
        elif self.task == "IAA":
            user_prompt_text = f"Considering its artistic composition, color harmony, and overall visual appeal, use an adjective to describe the aesthetic quality of this {media_type_str}?"
        else:
            raise ValueError(f"Unknown task: {self.task}")

        structured_user_content = [{"type": media_type_placeholder}, {"type": "text", "text": user_prompt_text}]
        conversation_for_sample = [{"role": "user", "content": structured_user_content}]
        
        prompt_base = self.processor.apply_chat_template(conversation_for_sample, tokenize=False, add_generation_prompt=True)
        final_prompt = prompt_base + f"The quality of the {media_type_str} is quite"
        
        current_image = media_data if not is_video else None
        current_video = media_data if is_video else None
        
        return final_prompt, current_image, current_video

    def forward(self, image_or_video_batch=None, labels=None, **args):
        batch_prompts, batch_images, batch_videos = [], [], []
        has_any_image, has_any_video = False, False

        for data_item in image_or_video_batch:
            prompt, image, video = self._prepare_prompt_and_media(data_item)
            batch_prompts.append(prompt)
            batch_images.append(image)
            batch_videos.append(video)
            if image is not None: has_any_image = True
            if video is not None: has_any_video = True

        images_arg = batch_images if has_any_image else None
        videos_arg = batch_videos if has_any_video else None
        
        inputs = self.processor(text=batch_prompts, images=images_arg, videos=videos_arg, padding=True, return_tensors="pt")
        
        device = self.model.device
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor): inputs[k] = v.to(device)
            elif isinstance(v, list): inputs[k] = [i.to(device) for i in v]
        
        return self.model(**inputs)

# =================================================================================================
# LLaVA NeXT Evaluator
# =================================================================================================
class LlavaNextEvaluator(nn.Module):
    def __init__(self, task="IQA", model_path="llava-hf/llava-v1.6-vicuna-7b-hf", local_files_only=False):
        super().__init__()
        self.task = task
        print(f"Loading LLaVA NeXT model from {model_path}")
        
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        if not torch.cuda.is_available(): dtype = torch.float32

        self.model = transformers.LlavaNextForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=dtype, device_map="auto", local_files_only=local_files_only,
            attn_implementation="sdpa"
        )
        self.processor = transformers.LlavaNextProcessor.from_pretrained(
            model_path, local_files_only=local_files_only
        )
        self.tokenizer = self.processor.tokenizer

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def _prepare_prompt_and_media(self, data_item):
        media_data = data_item["data"]
        # LLaVA NeXT (Image) usually expects single image
        if isinstance(media_data, list):
             # If video frames passed, take first one or handle as error? 
             # Assuming Image dataset here.
             media_data = media_data[0]
        
        media_type_str = "image"
        if self.task == "IQA":
            user_prompt_text = f"Taking into account the details and the rationality of the {media_type_str}, how would you rate the quality of this {media_type_str}?"
        elif self.task == "IAA":
            user_prompt_text = f"Considering its artistic composition, color harmony, and overall visual appeal, use an adjective to describe the aesthetic quality of this {media_type_str}?"
        else:
            raise ValueError(f"Unknown task: {self.task}")

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_prompt_text},
                ],
            },
        ]
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        prompt += f"The quality of the {media_type_str} is quite"
        
        return prompt, media_data

    def forward(self, image_or_video_batch=None, labels=None, **args):
        batch_prompts = []
        batch_images = []
        
        for data_item in image_or_video_batch:
            prompt, image = self._prepare_prompt_and_media(data_item)
            batch_prompts.append(prompt)
            batch_images.append(image)
            
        inputs = self.processor(text=batch_prompts, images=batch_images, padding=True, return_tensors="pt")
        
        device = self.model.device
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor): inputs[k] = v.to(device)
            
        return self.model(**inputs)
