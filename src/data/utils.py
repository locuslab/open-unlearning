import torch
import datasets
import numpy as np
import logging
from typing import List, Dict, Any, Union

IGNORE_INDEX = -100  # TODO put in common constants

logger = logging.getLogger("data")


def load_hf_dataset(path, **kwargs):
    dataset = datasets.load_dataset(path, **kwargs)
    return dataset


def preprocess_chat_instance(
    tokenizer,
    template_config: Dict[str, Any],
    prompt_msgs: Union[List[str], str],
    response_msgs: Union[List[str], str],
    max_length: int,
    predict_with_generate: bool = False,
) -> Dict[str, torch.Tensor]:
    """Preprocesses a chat instance for training or generation.
    When in training, both the returned `input_ids` and `labels` cover the entire conversation.
    `input_ids` has no padding, and `labels` assign `IGNORE_INDEX` to tokens where loss is not computed (i.e. all tokens except the final response message).
    When in generation, `input_ids` are returned only up to the last user prompt, excluding the assistant's response. The `labels` returned are the same as during training.
    `attention_mask` is always 1 over the full `input_ids` token sequence.

    `prompt_msgs` and `response_msgs` are lists where, except for the last pair, all
    corresponding pairs are in-context examples. When they are a string and not
    a list, there are no in-context examples.

    Args:
        tokenizer: Tokenizer to apply on text
        template_config (Dict[str, Any]): Configuration for the chat template (comes from model-specific config).
        prompt_msgs (Union[List[str], str]): List of prompt messages or a single prompt message string.
        response_msgs (Union[List[str], str]): List of response messages or a single response message string.
        max_length (int): Maximum sequence length after tokenization.
        predict_with_generate (bool, optional): Whether to prepare inputs for generation.

    Returns:
        Dict[str, torch.Tensor]: A dictionary containing 'input_ids', 'labels', and 'attention_mask' tensors for model input.
    """
    assert len(prompt_msgs) == len(response_msgs)
    if isinstance(prompt_msgs, str):
        assert isinstance(response_msgs, str)
        prompt_msgs, response_msgs = [prompt_msgs], [response_msgs]

    # Use .get() with default False to handle missing keys (e.g., empty template_args: {})
    apply_chat_template = template_config.get("apply_chat_template", False)
    if apply_chat_template:
        chat = []
        system_prompt = template_config.get("system_prompt", None)
        if system_prompt:
            chat += [{"role": "system", "content": system_prompt}]
        for prompt, response in zip(prompt_msgs, response_msgs):
            chat += [{"role": "user", "content": prompt}]
            chat += [{"role": "assistant", "content": response}]
        date_str = template_config.get("date_string", None)
        date_info = {"date_string": date_str} if date_str is not None else {}
        chat_ids = tokenizer.apply_chat_template(
            chat, tokenize=True, add_generation_prompt=False, **date_info
        )
        # all except last response are in-context examples
        wrapped_prompt = tokenizer.apply_chat_template(
            chat[:-1], tokenize=False, add_generation_prompt=True, **date_info
        )
        prompt_ids = tokenizer.apply_chat_template(
            chat[:-1], tokenize=True, add_generation_prompt=True, **date_info
        )
    else:
        wrapped_prompt = ""
        system_prompt_with_special_tokens = template_config.get(
            "system_prompt_with_special_tokens", None
        )
        if system_prompt_with_special_tokens:
            wrapped_prompt += system_prompt_with_special_tokens
        # add in-context examples
        n_few_shot = len(prompt_msgs) - 1
        # Use .get() with empty string defaults to handle missing keys
        user_start_tag = template_config.get("user_start_tag", "")
        user_end_tag = template_config.get("user_end_tag", "")
        asst_start_tag = template_config.get("asst_start_tag", "")
        asst_end_tag = template_config.get("asst_end_tag", "")
        for i in range(n_few_shot):
            fs_prompt, fs_response = prompt_msgs[i], response_msgs[i]
            wrapped_prompt += (
                user_start_tag
                + fs_prompt
                + user_end_tag
                + asst_start_tag
                + fs_response
                + asst_end_tag
            )

        # add actual example
        final_prompt, final_response = prompt_msgs[-1], response_msgs[-1]
        wrapped_prompt += (
            user_start_tag
            + final_prompt
            + user_end_tag
            + asst_start_tag
        )
        chat_ids = tokenizer(
            wrapped_prompt + final_response,
            add_special_tokens=True,
            max_length=max_length,
            truncation=True,
        )["input_ids"]

        prompt_ids = tokenizer(
            wrapped_prompt,
            add_special_tokens=True,
            max_length=max_length,
            truncation=True,
        )["input_ids"]

    if chat_ids[-1] != tokenizer.eos_token_id:
        chat_ids += [tokenizer.eos_token_id]

    len_matched = len(prompt_ids)

    item = {}
    if predict_with_generate:
        item["input_ids"] = prompt_ids
        labels = chat_ids  # contains the entire conversation
    else:
        item["input_ids"] = chat_ids
        labels = [IGNORE_INDEX] * len_matched + chat_ids[len_matched:]
        if len(prompt_ids) == len(chat_ids):
            # Rarely, tokenization can result in this condition being entered.
            # Say a input prompt is ABC and target output is D, tokenizer(ABCD)
            # can be [AB, CD] and tokenizer(ABC) can be [AB, C]. In this case,
            # we ignore loss on all indices in the labels. So, there is no way
            # to use this for next token prediction. Be careful while
            # interpreting results of such instances.
            logger.warning(
                "Tokenization mismatch: no valid target tokens for loss computation"
            )

    item["labels"] = labels
    item["attention_mask"] = [1] * len(item["input_ids"])
    for attr in item:
        item[attr] = torch.tensor(item[attr])
    return item


def preprocess_pretraining_instance(
    tokenizer,
    prefix: str,
    text_content: str,
    max_length: int,
    predict_with_generate: bool = False,
    insert_space: bool = False,
) -> Dict[str, torch.Tensor]:
    """Preprocesses a pretraining instance for training or generation.
    When in training, both the returned `input_ids` and `labels` are over the entire token sequence. `input_ids` has no padding, `labels` assigns `IGNORE_INDEX` to ignore all tokens that we don't compute loss over (i.e. the the 0th index token, all prefix tokens)
    When in generation, `input_ids` are returned only until the prefix portion. The `labels` returned are the same as during training.
    `attention_mask` is always 1 over the full input token sequence.
    Args:
        tokenizer: Tokenizer to apply on text
        prefix (str): The prefix string to prepend to the content.
        text_content (str): The main text content (following the prefix) to be tokenized.
        max_length (int): Maximum text content length after tokenization.
        predict_with_generate (bool, optional): Whether to prepare inputs for generation.
        insert_space (bool, optional): Whether to insert a space between prefix and content.

    Returns:
        Dict[str, torch.Tensor]: A dictionary containing 'input_ids', 'labels', and 'attention_mask' tensors for model input.
    """
    full_seq_ids = tokenizer(
        prefix + (" " if insert_space else "") + text_content, add_special_tokens=True
    )["input_ids"]
    prefix_ids = tokenizer(prefix, add_special_tokens=True)["input_ids"]
    prefix_len = len(prefix_ids)
    full_seq_ids = full_seq_ids[: prefix_len + max_length]  # manual truncation

    len_matched = prefix_len
    if len_matched == 0:  # never give loss on index 0, when prefix is empty
        len_matched = 1
    labels = [IGNORE_INDEX] * len_matched + full_seq_ids[len_matched:]
    item = {}
    if predict_with_generate:
        item["input_ids"] = prefix_ids
    else:
        item["input_ids"] = full_seq_ids
    item["labels"] = labels
    item["attention_mask"] = [1] * len(item["input_ids"])
    for attr in item:
        item[attr] = torch.tensor(item[attr])
    return item


def add_dataset_index(dataset):
    indexing = np.arange(len(dataset))
    dataset = dataset.add_column("index", indexing)
    return dataset
