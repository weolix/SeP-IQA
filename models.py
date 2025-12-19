import torch
import torch.nn as nn
import transformers
from PIL import Image

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
