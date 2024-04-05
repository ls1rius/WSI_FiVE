import torch
import torch.nn as nn
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, TaskType, get_peft_model, PeftModel, set_peft_model_state_dict, get_peft_model_state_dict
import os
import logging
logger = logging.getLogger(__name__)
class llama_text(nn.Module):
    """
    peft: lora
    """
    def __init__(self,
                 model_name_or_path,
                 tokenizer_path,
                 trainable="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj",
                 modules_to_save="embed_tokens,lm_head",
                 use_auth_token=False,
                 cache_dir=None,
                 model_revision="main",
                 lora_rank=8,
                 lora_dropout=0.05,
                 lora_alpha=32,
                 peft_path=None,
                 peft_sec_model=None):
        super().__init__()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
        self.torch_dtype = torch.float16
        # self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            from_tf=bool(".ckpt" in model_name_or_path),
            cache_dir=cache_dir,
            revision=model_revision,
            use_auth_token=True if use_auth_token else None,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
        )
        # # logger.info(f"len(tokenizer):{len(self.tokenizer)}")
        # embedding_size = self.model.get_input_embeddings().weight.shape[0]
        # # if len(self.tokenizer) != embedding_size:
        # if 49953 != embedding_size:
        #     logger.info("resize the embedding size by the size of the tokenizer")
        #     self.model.resize_token_embeddings(len(self.tokenizer))

        if peft_path is not None:
            logger.info("Peft from pre-trained model")
            model = PeftModel.from_pretrained(self.model, peft_path)
            if peft_sec_model is not None:
                adapters_weights = torch.load(
                    os.path.join(peft_path, peft_sec_model),
                    map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")
                )['module']
                set_peft_model_state_dict(model, adapters_weights)
        else:
            logger.info("Init new peft model")
            target_modules = trainable.split(',')
            if modules_to_save is not None:
                modules_to_save = modules_to_save.split(',')
            logger.info(f"target_modules: {target_modules}")
            logger.info(f"lora_rank: {lora_rank}")
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                target_modules=target_modules,
                inference_mode=False,
                r=lora_rank, lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                modules_to_save=modules_to_save)
            self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()
        logger.info(f"model.modules_to_save: {self.model.modules_to_save}")
        old_state_dict = self.model.state_dict
        self.model.state_dict = (
            lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
        ).__get__(self.model, type(self.model))
        self.project = nn.Linear(4096, 512, bias=False)

    def forward(self, x, labels=None):
        attention_mask = (x != 0)
        if labels is not None:
            with torch.no_grad():
                project_inverse = torch.linalg.pinv(self.model.base_model.get_output_embeddings().original_module.weight.T)
                output = self.model(input_ids=x, attention_mask=attention_mask, output_hidden_states=False)['logits'] @ project_inverse
            x_w_grad = self.model(input_ids=x[labels], attention_mask=attention_mask[labels], output_hidden_states=False)['logits'] @ project_inverse
            output[labels] = x_w_grad
        else:
            project_inverse = torch.linalg.pinv(self.model.base_model.get_output_embeddings().original_module.weight.T)
            output = self.model(input_ids=x, attention_mask=attention_mask, output_hidden_states=False)[
                         'logits'] @ project_inverse
        output = self.project(output)
        output = output.mean(dim=1)
        return output
