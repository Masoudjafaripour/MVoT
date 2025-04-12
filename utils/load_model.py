import requests
import torch
import math

# from PIL import Image

from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image

# from model_utils.wrapped_visualizer import KosmosVforConditionalGeneration

def load_model(args):
    model_name = args.model

    model_ckpt_path = args.model_ckpt

    if model_name in ['anole']:
        image_token_num = args.image_seq_length

        from model_utils.wrapped_visualizer import AnoleforConditionalGeneration
        model = AnoleforConditionalGeneration.from_pretrained(
            "leloy/Anole-7b-v0.1-hf",
            device_map="cuda",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            codebook_sim="mse"
        )
        processor = AutoProcessor.from_pretrained("leloy/Anole-7b-v0.1-hf", image_seq_length=image_token_num)
        processor.image_processor.size = {"shortest_edge": int(512 / int(math.sqrt(1024 / image_token_num)))}
        processor.image_processor.crop_size = {
            "height": int(512 / int(math.sqrt(1024 / image_token_num))),
            "width": int(512 / int(math.sqrt(1024 / image_token_num)))
        }

        model.config.pad_token_id = processor.tokenizer.pad_token_id
        model.model.vqmodel.config.resolution = processor.image_processor.size["shortest_edge"]
        model.model.vqmodel.quantize.quant_state_dims = [
            model.model.vqmodel.config.resolution // 2 ** (len(model.model.vqmodel.config.channel_multiplier) - 1)
        ] * 2

        args.sketch_resolution = model.model.vqmodel.config.resolution
        model.sketch_resolution = (args.sketch_resolution, args.sketch_resolution)
        model.image_token_num = image_token_num

        model.get_vis_codebook_sim()

        from peft import LoraConfig, get_peft_model
        from peft.peft_model import PeftModel

        config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=['q_proj', "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.1,
            bias="none",
            modules_to_save=["lm_head"],
        )
        lora_model = get_peft_model(model, config)

        if args.do_eval and not args.do_train and model_ckpt_path:
            lora_model.load_adapter(model_ckpt_path, 'default', is_trainable=False)

        return {
            'processor': processor,
            'model': lora_model
        }
    else:
        raise ValueError("Unsupported model type. ")