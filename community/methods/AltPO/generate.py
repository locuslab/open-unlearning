import os
import json
import logging
from typing import List

import hydra
import transformers
from datasets import load_dataset
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from log_config import setup_logging_config
from prompts import (
    INST_QAS_INSTR,
    INST_QAS_TEMPLATE,
    INST_QAS_TEMPLATE_QUERY,
    INST_QAS_LLAMA3_INSTR,
    INST_QAS_LLAMA3_TEMPLATE,
    INST_QAS_LLAMA3_TEMPLATE_QUERY,
)

# Setup logger
logger = logging.getLogger(__name__)


HF_HOME = os.getenv('HF_HOME', '~/.cache/huggingface')

def get_model(config):
    # Check if the model name is in the predefined list or not
    model_name = config['model_kwargs']['pretrained_model_name_or_path']
    logger.info(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(**config['model_kwargs'], cache_dir=HF_HOME)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Setting the padding token for tokenizer (has to be done manually - not taken care by Hugging face library)
    if tokenizer.pad_token:
        logger.debug("Using existing pad_token")
    elif tokenizer.unk_token:
        tokenizer.pad_token_id = tokenizer.unk_token_id
        logger.debug("Set pad_token_id to unk_token_id")
    elif tokenizer.eos_token:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        logger.debug("Set pad_token_id to eos_token_id")
    else:
        raise ValueError("Unable to set the pad token for the model")
    logger.info("Model and tokenizer loaded successfully")
    # returns model and the tokenizer
    return model, tokenizer


def get_dataset(config):
    dataset_name = config['dataset_name']
    logger.info(f"Loading dataset: {dataset_name}")
    if dataset_name == 'tofu':
        dataset = load_dataset(**config['dataset_kwargs'])
    else:
        raise ValueError(f"Dataset '{dataset_name}' not implemented")
    def add_numbering(example, idx):
        example_new = {}
        example_new['doc_id'] = idx
        example_new.update(example)
        return example_new
    # return the loaded dataset (stored in cache)
    numbered_dataset = dataset.map(add_numbering, with_indices=True)
    logger.info(f"Dataset loaded successfully. Total examples: {len(numbered_dataset)}")
    return numbered_dataset

def read_json(path):
        with open(path) as json_file:
            json_data = json.load(json_file)
        return json_data

def aggregate_fewshot(prompts, prompt_query, **kwargs):
    fewshot_delimiter = kwargs.get("fewshot_delimiter", "\n\n")
    aggregated_prompt = fewshot_delimiter.join(prompts+[prompt_query])
    return aggregated_prompt

def get_prompts(config):
    # returns the constant defined in the src
    prompt_name = config['prompt_name']
    logger.info(f"Using prompt template: {prompt_name}")
    if prompt_name == 'INST_QAS_TEMPLATE':
        prompt_template = INST_QAS_TEMPLATE
        prompt_template_query = INST_QAS_TEMPLATE_QUERY
    elif prompt_name == 'INST_QAS_LLAMA3_TEMPLATE':
        prompt_template = INST_QAS_LLAMA3_TEMPLATE
        prompt_template_query = INST_QAS_LLAMA3_TEMPLATE_QUERY 
    else:
        raise NotImplementedError(f"Prompt template '{prompt_name}' not implemented")

    examples_path = config.get('examples_path', None)
    if examples_path is None:
        examples = []
        logger.debug("No examples path provided, using zero-shot")
    else:
        logger.info(f"Loading examples from: {examples_path}")
        examples = read_json(examples_path)
        n_shot = config.get("n_shot", len(examples))
        examples = examples[:n_shot]
        logger.info(f"Using {len(examples)} examples for few-shot prompting")
    prompts = [prompt_template.format(**example) for example in examples]
    aggregated_template = aggregate_fewshot(prompts, prompt_template_query,**config)
    return aggregated_template

    
# Question filling in the prompts
def prompt_infilling_batch(batch, prompt, **kwargs):
    inputs = []
    keys = list(batch.keys())
    for i in range(len(batch[keys[0]])):
        example = {k:batch[k][i] for k in keys}
        inputs.append(custom_format(prompt, {**example, **kwargs}))
    return inputs

def custom_format(prompt, example):
    for k,v in example.items():
        substring = "{" + k + "}"
        prompt = prompt.replace(substring, str(v))
    return prompt

def tok_batch_encode(
        strings,
        tokenizer,
        padding_side: str = "left",
        left_truncate_len: int = None,
        truncation: bool = False,
    ):
    # encode a batch of strings. converts to tensors and pads automatically, unlike tok_encode.
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = padding_side
    encoding = tokenizer(
        strings,
        truncation=truncation,
        padding="longest",
        return_tensors="pt",
    )
    # TODO: handle differently for gemma models , we need to add bos_token
    
    if left_truncate_len:
        encoding["input_ids"] = encoding["input_ids"][:, -left_truncate_len:]
        encoding["attention_mask"] = encoding["attention_mask"][
            :, -left_truncate_len:
        ]
    tokenizer.padding_side = old_padding_side

    return encoding["input_ids"], encoding["attention_mask"]

# Decoding model output tokens
def tok_decode(tokens, tokenizer):
    return tokenizer.decode(tokens, skip_special_tokens=True)

class MultiTokenEOSCriteria(transformers.StoppingCriteria):
    """Criteria to stop on the specified multi-token sequence."""

    def __init__(
        self,
        sequence: str,
        tokenizer: transformers.PreTrainedTokenizer,
        initial_decoder_input_length: int,
        batch_size: int,
    ) -> None:
        self.initial_decoder_input_length = initial_decoder_input_length
        self.done_tracker = [False] * batch_size
        self.sequence = sequence
        self.sequence_ids = tokenizer.encode(sequence, add_special_tokens=False)
        # print(sequence, self.sequence_ids)
        # we look back for 2 more tokens than it takes to encode our stop sequence
        # because tokenizers suck, and a model might generate `['\n', '\n']` but our `sequence` is `['\n\n']`
        # and we don't want to mistakenly not stop a generation because our
        # (string) stop sequence was output in a different tokenization

        # NOTE: there is a minor danger that this will end up looking back 2 tokens into the past, into the inputs to the model,
        # and stopping generation immediately as a result. With only 2 extra tokens of lookback, this risk is minimized
        # Additionally, in lookback_ids_batch we should prevent ever looking back into the inputs as described.
        self.sequence_id_len = len(self.sequence_ids) + 2
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        # For efficiency, we compare the last n tokens where n is the number of tokens in the stop_sequence
        lookback_ids_batch = input_ids[:, self.initial_decoder_input_length :]

        lookback_ids_batch = lookback_ids_batch[:, -self.sequence_id_len :]

        lookback_tokens_batch = self.tokenizer.batch_decode(lookback_ids_batch)

        for i, done in enumerate(self.done_tracker):
            if not done:
                self.done_tracker[i] = self.sequence in lookback_tokens_batch[i]
        return False not in self.done_tracker


def stop_sequences_criteria(
    tokenizer: transformers.PreTrainedTokenizer,
    stop_sequences: List[str],
    initial_decoder_input_length: int,
    batch_size: int,
) -> transformers.StoppingCriteriaList:
    return transformers.StoppingCriteriaList(
        [
            *[
                MultiTokenEOSCriteria(
                    sequence, tokenizer, initial_decoder_input_length, batch_size
                )
                for sequence in stop_sequences
            ],
        ]
    )


def collate_fn(batch):
    return {key: [i[key] for i in batch] for key in batch[0]}


@hydra.main(version_base=None, config_path=".", config_name="generate")
def main(config):
    # Setup logging - use Hydra's output directory if available
    try:
        hydra_cfg = GlobalHydra.instance().hydra
        output_dir = hydra_cfg.runtime.output_dir
        log_file = f"{output_dir}/generate.log"
    except:
        log_file = None
    setup_logging_config(log_file=log_file)
    
    logger.info("=" * 80)
    logger.info("Starting generation process")
    logger.info("=" * 80)
    
    seed = config.get('seed', 0)
    logger.info(f"Setting random seed: {seed}")
    set_seed(seed)
    
    # Loading the model and the tokenizer
    model, tokenizer = get_model(config['model_config'])
    
    # Load the dataset - a list of dictionary - question and answer pairs
    dataset = get_dataset(config['dataset_config'])
    
    # Having prompts defined for the model input 
    prompt = get_prompts(config['prompt_config'])
    suff = ""
    if '-+' in prompt:
        prompt_num = prompt.split('-+')[-1][0]
        split_symbol = f'-+{prompt_num}'
        suff = f'_v{prompt_num}'
        prompt = prompt.replace(split_symbol, '')
        logger.debug(f"Processed prompt suffix: {suff}")
    
    # get the outdir
    outdir = config.get("outdir", "outdir")
    limit = config.get('limit', None)
    if limit:
        # If there is a limit, we select top n questions from the dataset
        logger.info(f"Limiting dataset to {limit} examples")
        dataset = dataset.select(range(limit))
    
    batch_size = config['batch_size']
    logger.info(f"Batch size: {batch_size}")
    
    # Loading the question answer pairs from the dataset
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    logger.info(f"DataLoader created with {len(data_loader)} batches")
    
    # Padding and truncation
    left_truncate_len = config.get('left_truncate_len', None)
    padding_side = config.get('padding_side', 'left')
    truncation = config.get('truncation', False)
    logger.debug(f"Tokenization settings: padding_side={padding_side}, truncation={truncation}, left_truncate_len={left_truncate_len}")
    
    # self consistency
    repeats = config.get('repeats', 1)
    if repeats > 1:
        logger.info(f"Using self-consistency with {repeats} repeats")
    
    # Get all the different parameters used in model.generate() function
    generation_kwargs = config.get('generation_kwargs', {})
    # Convert to dict if it's a DictConfig
    try:
        if isinstance(generation_kwargs, OmegaConf):
            generation_kwargs = OmegaConf.to_container(generation_kwargs, resolve=True)
    except:
        logger.warning("generation_kwargs is not a DictConfig, converting to dict")
    if not isinstance(generation_kwargs, dict):
        generation_kwargs = {}
    
    # Remove sampling-related parameters if do_sample is False to avoid warnings
    if not generation_kwargs.get("do_sample", False):
        removed_params = []
        for key in ["top_k", "top_p", "temperature"]:
            if key in generation_kwargs:
                generation_kwargs.pop(key, None)
                removed_params.append(key)
        if removed_params:
            logger.debug(f"Removed sampling parameters (do_sample=False): {removed_params}")
    
    logger.info(f"Generation kwargs: {generation_kwargs}")
    
    # These are the list of tokens that stops further generation when encountered 
    until = config.get('until', [])
    if until:
        logger.info(f"Stop sequences: {until}")
    
    device = config.get('device')
    logger.info(f"Using device: {device}")
    
    results = []
    logger.info("Starting generation loop...")
    # a batch is generated. It is a dictionary having question and answer as keys, and corresponding values as a list
    for batch_idx, batch in enumerate(tqdm(data_loader, desc="Generating")):
        # Replacing the variables with actual values/questions in prompt
        inputs = prompt_infilling_batch(batch, prompt)
        input_ids, attention_mask = tok_batch_encode(inputs, tokenizer, padding_side, left_truncate_len, truncation)
        res_sc = []
        for repeat in range(repeats):
            # Create a stopping criteria - stop generation once encounters a token specified in the list
            stopping_criteria = stop_sequences_criteria(
                tokenizer, until, input_ids.shape[1], input_ids.shape[0]
            )
            # Generating tokens
            output = model.generate(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
                stopping_criteria=stopping_criteria,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
                **generation_kwargs,
            )
            out_toks_list = output.tolist()
            res = []
            for cont_toks in out_toks_list:
                cont_toks = cont_toks[input_ids.shape[1] :]
                s = tok_decode(cont_toks, tokenizer)
                for term in until:
                    if len(term) > 0:
                        s = s.split(term)[0]
                s = s.strip()
                res.append(s)
            batch.update({'input':inputs, "sub_answer":res})
            res_sc += [{k: v[i] for k, v in batch.items()} for i in range(len(list(batch.values())[0]))]
        results += res_sc
        if (batch_idx + 1) % 10 == 0:
            logger.debug(f"Processed {batch_idx + 1} batches, {len(results)} total results")
    
    output_file = config.output_file
    outdir = os.path.dirname(output_file)
    logger.info(f"Saving results to: {output_file}")
    # Write the list to a JSON file
    os.makedirs(outdir, exist_ok=True)
    # r_dump = []
    with open(output_file, 'w') as f:
        for result in results:
            r = {
                "question": result["question"],
                "answer": result["answer"],
                "alternate": result["sub_answer"]
            }
            # r = result
            json.dump(r, f)
            f.write('\n')
    
    logger.info(f"Generation completed successfully. Generated {len(results)} results")
    logger.info("=" * 80)
    
if __name__ == '__main__':
    main()
    