import torch
from torch.utils.data import Dataset

from data.utils import (
    IGNORE_INDEX,
    load_hf_dataset,
    preprocess_chat_instance,
    add_dataset_index,
)


class QADataset(Dataset):
    def __init__(
        self,
        hf_args,
        template_args,
        tokenizer,
        question_key="question",
        answer_key="answer",
        few_shot_dataset_hf_args=None,
        max_length=512,
        predict_with_generate=False,
        popularity_key=None,
    ):
        super(QADataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = load_hf_dataset(**hf_args)
        self.data = add_dataset_index(self.data)
        self.fs_data = None
        if few_shot_dataset_hf_args is not None:
            raw_data = load_hf_dataset(**few_shot_dataset_hf_args)
            self.fs_data = {}
            self.fs_data[question_key] = raw_data[question_key]
            self.fs_data[answer_key] = raw_data[answer_key]
        self.template_args = template_args
        self.question_key = question_key
        self.answer_key = answer_key
        self.predict_with_generate = predict_with_generate
        # optional numeric column returned as `popularity` (used by AdaPop)
        self.popularity_key = popularity_key
        if popularity_key is not None:
            if popularity_key not in self.data.column_names:
                raise ValueError(
                    f"popularity_key '{popularity_key}' is not a column of the dataset, "
                    f"available columns: {self.data.column_names}"
                )
            dtype = getattr(self.data.features[popularity_key], "dtype", None)
            if not str(dtype).startswith(("int", "uint", "float")):
                raise ValueError(
                    f"popularity column '{popularity_key}' must be int or float, got {dtype}"
                )

    def __len__(self):
        return len(self.data)

    def _process_sample(self, question, answer, index=-1):
        if self.fs_data is None:
            prompt_msgs, response_msgs = [question], [answer]
        else:
            prompt_msgs = self.fs_data[self.question_key] + [question]
            response_msgs = self.fs_data[self.answer_key] + [answer]
        tokenized_data = preprocess_chat_instance(
            self.tokenizer,
            self.template_args,
            prompt_msgs,
            response_msgs,
            self.max_length,
            self.predict_with_generate,
        )
        item_dct = {
            "input_ids": tokenized_data["input_ids"],
            "labels": tokenized_data["labels"],
            "attention_mask": tokenized_data["attention_mask"],
            "index": index,
        }
        return item_dct

    def __getitem__(self, idx):
        question = self.data[idx][self.question_key]
        answer = self.data[idx][self.answer_key]
        index = self.data[idx]["index"]
        if isinstance(answer, str):
            item = self._process_sample(question=question, answer=answer, index=index)
        elif isinstance(answer, list):
            item = {}
            for i, ans in enumerate(answer):
                sample_item = self._process_sample(
                    question=question, answer=ans, index=index
                )
                item[i] = sample_item
        else:
            raise NotImplementedError("answer format not found")
        if self.popularity_key is not None and isinstance(answer, str):
            item["popularity"] = float(self.data[idx][self.popularity_key])
        return item


class QAwithIdkDataset(QADataset):
    def __init__(self, idk_path, return_original=True, *args, **kwargs):
        self.idk_path = idk_path
        self.return_original = return_original
        self.idk_responses = open(self.idk_path, "r").readlines()
        super().__init__(*args, **kwargs)

    def item_with_idk(self, question):
        rand_pos = torch.randint(0, len(self.idk_responses), (1,)).item()
        idk_response = self.idk_responses[rand_pos].strip()
        idk_item = self._process_sample(question=question, answer=idk_response)
        return idk_item

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        question = self.data[idx][self.question_key]
        if isinstance(item, dict):
            return_item = {"original": item}
            idk_item = self.item_with_idk(question)
            return_item["alternate"] = idk_item
            # return_item = [item, idk_item]
        elif isinstance(item, list) or isinstance(item, tuple):
            return_item = []
            for sample_item in item:
                return_item = {"original": sample_item}
                idk_item = self.item_with_idk(question)
                return_item["alternate"] = idk_item
                # return_item.append([sample_item, idk_item])
        return return_item if self.return_original else return_item["alternate"]


class QAwithAlternateDataset(QADataset):
    def __init__(self, alternate_key, return_original=True, *args, **kwargs):
        self.alternate_key = alternate_key
        self.return_original = return_original
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        question = self.data[idx][self.question_key]
        if isinstance(item, dict):
            return_item = {"original": item}
            alt_item = self._process_sample(
                question=question, answer=self.data[idx][self.alternate_key]
            )
            return_item["alternate"] = alt_item
            # return_item = [item, idk_item]
        elif isinstance(item, list) or isinstance(item, tuple):
            return_item = []
            for sample_item in item:
                return_item = {"original": sample_item}
                alt_item = self._process_sample(
                    question=question, answer=self.data[idx][self.alternate_key]
                )
                return_item["alternate"] = alt_item
                # return_item.append([sample_item, idk_item])
        return return_item if self.return_original else return_item["alternate"]


class QAwithEntityDataset(Dataset):
    """QA pairs whose answers are annotated as `prefix + continuation`, with an entity
    inside the continuation, e.g. the UDS-annotated TOFU forget set
    (https://arxiv.org/abs/2605.24614). As in the reference UDS implementation, the
    prompt is the raw text `Question: {question}` + newline + `Answer: {prefix}`, and
    the continuation is tokenized separately and appended. `template_args` is not used,
    to keep scores comparable with the paper. Only the entity tokens are labelled, so
    metrics see teacher-forced predictions of the entity alone."""

    def __init__(
        self,
        hf_args,
        template_args,
        tokenizer,
        question_key="question",
        answer_key="answer",
        prefix_key="prefix",
        entity_key="entity",
        index_key="idx",
        max_length=512,
    ):
        super(QAwithEntityDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = load_hf_dataset(**hf_args)
        if index_key not in self.data.column_names:
            self.data = add_dataset_index(self.data)
            index_key = "index"
        self.question_key = question_key
        self.answer_key = answer_key
        self.prefix_key = prefix_key
        self.entity_key = entity_key
        self.index_key = index_key

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prefix, answer = item[self.prefix_key], item[self.answer_key]
        entity = item[self.entity_key]
        assert answer.startswith(prefix), f"prefix not found in answer {idx}"
        prompt = f"Question: {item[self.question_key]}\nAnswer: {prefix}"
        continuation = answer[len(prefix) :]
        if not continuation[:1].isspace() and not prompt[-1].isspace():
            continuation = " " + continuation  # keep a word boundary, as in the paper
        entity_start = continuation.find(entity)
        assert entity_start >= 0, f"entity not found in answer {idx}"
        entity_end = entity_start + len(entity)

        prompt_ids = self.tokenizer(prompt, add_special_tokens=True)["input_ids"]
        tokenized = self.tokenizer(
            continuation, add_special_tokens=False, return_offsets_mapping=True
        )
        input_ids = (prompt_ids + tokenized["input_ids"])[: self.max_length]
        labels = [IGNORE_INDEX] * len(input_ids)
        for i, (start, end) in enumerate(tokenized["offset_mapping"]):
            pos = len(prompt_ids) + i
            if end > entity_start and start < entity_end and pos < len(input_ids):
                labels[pos] = input_ids[pos]
        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
            "index": item[self.index_key],
        }
