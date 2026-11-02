from copy import deepcopy
import torch
import transformers
from peft import PeftModel
import os
from pathlib import Path
import dataclasses
from enum import Enum
from typing import Union, List, Dict
import sys

sys.path.append("..")
from utils import attack_utility
from . import config

DEFENDED_MODEL_COMMON_PATH = os.environ.get(
    "ABC_MODEL_DIR", str(Path(__file__).resolve().parent / "secalign_models")
)
MODEL_REL_PATHS = {

    ("mistralai", "undefended"): 'mistralai/Mistral-7B-v0.1_SpclSpclSpcl_None_2025-03-12-01-02-08',

}

def load_model_and_tokenizer(model_path, tokenizer_path=None, device="cuda:0", **kwargs):
    model = (
        transformers.AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True, **kwargs
        )
        # .to(device)
        .eval()
    )
    tokenizer_path = model_path if tokenizer_path is None else tokenizer_path
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    # 针对特定模型做 tokenizer 调整
    if "oasst-sft-6-llama-30b" in tokenizer_path:
        tokenizer.bos_token_id = 1
        tokenizer.unk_token_id = 0
    if "guanaco" in tokenizer_path:
        tokenizer.eos_token_id = 2
        tokenizer.unk_token_id = 0
    if "llama-2" in tokenizer_path:
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.padding_side = "left"
    if "falcon" in tokenizer_path:
        tokenizer.padding_side = "left"
    if "mistral" in tokenizer_path:
        tokenizer.padding_side = "left"
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

def _form_chat_template_from_frontend_delimiters(frontend_delimiters):
    formatter = config.PROMPT_FORMAT[frontend_delimiters]["prompt_input"]
    return (
        formatter
        .replace("{instruction}", "{{ messages[0]['content'] }}")
        .replace("{input}", "{{ messages[1]['content'] }}")
    )

def load_lora_model(model_name_or_path, device='0', load_model=True, **kwargs):
    configs = model_name_or_path.split('/')[-1].split('_') + ['Frontend-Delimiter-Placeholder', 'None']
    # 一般模型命名为 基座模型名_对齐方式标记_其他说明
    # 如：llama2_dpo_instruct
    for alignment in ['dpo', 'kto', 'orpo']:
        base_model_index = model_name_or_path.find(alignment) - 1
        if base_model_index > 0: break
        else: base_model_index = False

    base_model_path = model_name_or_path[:base_model_index] if base_model_index else model_name_or_path
    frontend_delimiters = configs[1] if configs[1] in config.DELIMITERS else base_model_path.split('/')[-1]
    training_attacks = configs[2]
    if not load_model: return base_model_path, None, frontend_delimiters, None

    print("模型地址为", base_model_path)

    model, tokenizer = load_model_and_tokenizer(base_model_path, low_cpu_mem_usage=True, use_cache=False, device="cuda:" + device, **kwargs)

    try:
        _ = tokenizer.apply_chat_template([
            {
                "role": "system",
                "content": "ABC"
            },
            {
                "role": "user",
                "content": "DEF"
            }
        ])
    except Exception:
        print("缺少默认的 chat 模板")
        tokenizer.chat_template = _form_chat_template_from_frontend_delimiters(frontend_delimiters)
        _ = tokenizer.apply_chat_template([
            {
                "role": "system",
                "content": "ABC"
            },
            {
                "role": "user",
                "content": "DEF"
            }
        ])

    if 'Instruct' in model_name_or_path: tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = 512
    if base_model_index: model = PeftModel.from_pretrained(model, model_name_or_path, is_trainable=False)
    return model, tokenizer, frontend_delimiters, training_attacks


def maybe_load_secalign_defended_model(model_name, defence, **kwargs):
    """Load the local target model used by ABC."""
    if defence != "undefended":
        raise ValueError("Only undefended target models are included in this release.")
    relative_path = MODEL_REL_PATHS.get((model_name, defence))
    model_path = os.path.join(DEFENDED_MODEL_COMMON_PATH, relative_path) if relative_path else model_name
    return load_lora_model(model_path, **kwargs)


def _convert_to_secalign_format(
    input_conv,
    prompt_template,
    tokenizer,
    harmful_inst
):
    assert isinstance(input_conv, list) and all([isinstance(conv_part, dict) for conv_part in input_conv])
    inst_str = deepcopy(input_conv[0]["content"])
    data_str = deepcopy(input_conv[1]["content"])
    data_str += " " + attack_utility.ADV_PREFIX_INDICATOR + " " + harmful_inst + " " + attack_utility.ADV_SUFFIX_INDICATOR + " "
    static_string = prompt_template.format_map({"instruction": inst_str, "input": data_str})
    # print("加入触发器之后的样本：\n",static_string)
    input_conv = tokenizer.batch_decode(tokenizer([static_string])["input_ids"], clean_up_tokenization_spaces=False)[0]
    return input_conv
