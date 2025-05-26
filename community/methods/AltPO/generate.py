import os
import json
import hydra
import transformers
from tqdm import tqdm
from typing import List
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

# Llama2
INST_QAS_INSTR = "Now write another version of the answer with some alternate plausible facts that change answer details.\n"
INST_QAS_TEMPLATE= """[INST] Question: {question}\nAnswer: {answer}\n"""+INST_QAS_INSTR + """ [/INST]"""+""" Alternate Answer :{sub_answer}"""
INST_QAS_TEMPLATE_QUERY = """[INST] Question: {question}\nAnswer:{answer}\n"""+INST_QAS_INSTR + """ [/INST]""" + """ Alternate Answer :"""

# Llama3.2
INST_QAS_LLAMA3_INSTR = "Now pretend you are making things up. Write another answer to the question that is of a different template than the given answer and changes all facts from what are introduced in the given answer (changed answers must be plausible while being inconsistent with given answer). Ensure that your alternate answer is a plausible response to the question and doesn't change any details mentioned in question and only introduces changes to all the facts introduced answer."
INST_QAS_LLAMA3_TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful AI assistant<|eot_id|>
<|start_header_id|>user<|end_header_id|>

Question: {question}\nAnswer: {answer}\n"""+INST_QAS_LLAMA3_INSTR+"""<|eot_id|>
<|start_header_id|>assistant<|end_header_id|> 

"""+"""Alternate Answer :{sub_answer}""" + """<|eot_id|>"""
INST_QAS_LLAMA3_TEMPLATE_QUERY = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful AI assistant<|eot_id|>
<|start_header_id|>user<|end_header_id|>

Question: {question}\nAnswer: {answer}\n"""+INST_QAS_LLAMA3_INSTR+"""<|eot_id|>
<|start_header_id|>assistant<|end_header_id|> 

"""+"""Alternate Answer :"""


HF_HOME = os.getenv('HF_HOME', '~/.cache/huggingface')

def get_model(config):
    # Check if the model name is in the predefined list or not
    model = AutoModelForCausalLM.from_pretrained(**config['model_kwargs'], cache_dir=HF_HOME)
    tokenizer = AutoTokenizer.from_pretrained(config['model_kwargs']['pretrained_model_name_or_path'])
    
    # Setting the padding token for tokenizer (has to be done manually - not taken care by Hugging face library)
    if tokenizer.pad_token:
        pass
    elif tokenizer.unk_token:
        tokenizer.pad_token_id = tokenizer.unk_token_id
    elif tokenizer.eos_token:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    else:
        raise ValueError("Unable to set the pad token for the model")
    # returns model and the tokenizer
    return model, tokenizer


def get_dataset(config):
    if config['dataset_name'] == 'tofu':
        dataset = load_dataset(**config['dataset_kwargs'])
    else:
        raise ValueError("dataset not implemented")
    def add_numbering(example, idx):
        example_new = {}
        example_new['doc_id'] = idx
        example_new.update(example)
        return example_new
    # return the loaded dataset (stored in cache)
    numbered_dataset = dataset.map(add_numbering, with_indices=True)
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
    if config['prompt_name'] == 'INST_QAS_TEMPLATE':
        prompt_template = INST_QAS_TEMPLATE
        prompt_template_query = INST_QAS_TEMPLATE_QUERY
    elif config['prompt_name'] == 'INST_QAS_LLAMA3_TEMPLATE':
        prompt_template = INST_QAS_LLAMA3_TEMPLATE
        prompt_template_query = INST_QAS_LLAMA3_TEMPLATE_QUERY 
    else:
        raise NotImplementedError

    examples_path = config.get('examples_path', None)
    if examples_path is None:
        examples = []
    else:
        examples = read_json(examples_path)
        examples = examples[:config.get("n_shot", len(examples))]
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
    set_seed(config.get('seed', 0))
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
    # get the outdir
    outdir = config.get("outdir", "outdir")
    limit = config.get('limit', None)
    if limit:
        # If there is a limit, we select top n questions from the dataset
        dataset = dataset.select(range(limit))
    # Loading the question answer pairs from the dataset
    data_loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False, collate_fn= collate_fn)
    # Padding and truncation
    left_truncate_len = config.get('left_truncate_len', None)
    padding_side = config.get('padding_side', 'left')
    truncation = config.get('truncation', False)
    # self consistency
    repeats = config.get('repeats', 1)
    # Get all the different parameters used in model.generate() function
    generation_kwargs = config.get('generation_kwargs')
    # These are the list of tokens that stops further generation when encountered 
    until = config.get('until', [])
    device = config.get('device')
    results = []
    # a batch is generated. It is a dictionary having question and answer as keys, and corresponding values as a list
    for batch in tqdm(data_loader):
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
    outdir = os.path.dirname(config.output_file)
    # Write the list to a JSON file
    os.makedirs(outdir, exist_ok=True)
    # r_dump = []
    with open(config.output_file, 'w') as f:
        for result in results:
            r = {
                "question": result["question"],
                "answer": result["answer"],
                "alternate": result["sub_answer"]
            }
            # r = result
            json.dump(r, f)
            f.write('\n')
    
if __name__ == '__main__':
    main()
    