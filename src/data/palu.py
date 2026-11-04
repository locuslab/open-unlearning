"""Load PALU annotations and map answer character spans to token labels."""

import json

import torch
from datasets import load_from_disk
from torch.utils.data import Dataset


class PALUDataset(Dataset):
    """Prepare annotated forget examples for PALU using a fast tokenizer.

    Labels are token IDs for sensitive answer tokens, -200 for other answer
    tokens, and -100 for the prompt. Padding is handled by the existing collator.

    Args:
        data_path: Directory produced by Hugging Face Dataset.save_to_disk.
        template_args: The model's existing chat or manual prompt template.
        tokenizer: Fast tokenizer providing character offset mappings.
        question_key: Question column in the annotated dataset.
        answer_key: Answer column in the annotated dataset.
        target_key: Column containing sensitive character spans, optionally JSON.
        max_length: Maximum sequence length, including any appended EOS token.
    """

    def __init__(
        self,
        data_path,
        template_args,
        tokenizer,
        question_key="question",
        answer_key="answer",
        target_key="target_words",
        max_length=512,
    ):
        if not tokenizer.is_fast:
            raise ValueError("PALUDataset requires a fast tokenizer for span offsets")
        if max_length < 2:
            raise ValueError("max_length must be at least 2")
        self.data = load_from_disk(data_path)
        required = {question_key, answer_key, target_key}
        if not required.issubset(self.data.column_names):
            raise ValueError(f"PALUDataset requires columns: {sorted(required)}")
        self.template_args = template_args
        self.tokenizer = tokenizer
        self.question_key = question_key
        self.answer_key = answer_key
        self.target_key = target_key
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """Tokenize one QA pair and label only tokens overlapping annotated spans."""
        row = self.data[idx]
        question, answer = row[self.question_key], row[self.answer_key]
        spans = row[self.target_key]
        if isinstance(spans, str):
            spans = json.loads(spans)
        spans = spans or []
        for span in spans:
            if not 0 <= span["start"] < span["end"] <= len(answer):
                raise ValueError(
                    f"Invalid sensitive span in PALU example {idx}: {span}"
                )

        template = self.template_args
        use_chat_template = template["apply_chat_template"]
        if use_chat_template:
            messages = []
            if template.get("system_prompt"):
                messages.append(
                    {"role": "system", "content": template["system_prompt"]}
                )
            messages.append({"role": "user", "content": question})
            date = template.get("date_string")
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                **({"date_string": date} if date is not None else {}),
            )
        else:
            prompt = (
                (template.get("system_prompt_with_special_tokens") or "")
                + template["user_start_tag"]
                + question
                + template["user_end_tag"]
                + template["asst_start_tag"]
            )

        encoded = self.tokenizer(
            prompt + answer,
            add_special_tokens=not use_chat_template,
            truncation=True,
            max_length=self.max_length,
            return_offsets_mapping=True,
        )
        input_ids = encoded["input_ids"]
        labels = []
        # Native offsets also cover the final token and tokens crossing the prompt boundary.
        for token_id, (start, end) in zip(input_ids, encoded["offset_mapping"]):
            if end <= len(prompt) or start == end:
                labels.append(-100)
            elif any(
                start < len(prompt) + span["end"] and end > len(prompt) + span["start"]
                for span in spans
            ):
                labels.append(token_id)
            else:
                labels.append(-200)
        eos = self.tokenizer.eos_token_id
        if eos is not None and len(input_ids) < self.max_length:
            if not input_ids or input_ids[-1] != eos:
                input_ids.append(eos)
                labels.append(-200)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "index": idx,
        }
