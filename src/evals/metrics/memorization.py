import logging
import torch
import numpy as np
import scipy as sc
from torch.utils.data import DataLoader


from evals.metrics.utils import (
    aggregate_to_1D,
    evaluate_probability,
    eval_text_similarity,
    run_batchwise_evals,
    tokenwise_vocab_logprobs,
)
from evals.metrics.base import unlearning_metric

# Supress the info messages logged while calculating rouge using rouge_scorer
logging.getLogger("absl").setLevel(logging.WARNING)


@unlearning_metric(name="probability")
def probability(model, **kwargs):
    """Compute the probabilities by data points and report aggregated average"""
    data = kwargs["data"]
    collator = kwargs["collators"]
    batch_size = kwargs["batch_size"]

    dataloader = DataLoader(data, batch_size=batch_size, collate_fn=collator)

    fun_args = {}
    scores_by_index = run_batchwise_evals(
        model, dataloader, evaluate_probability, fun_args, "Calculating loss"
    )
    prob_values = np.array([evals["prob"] for evals in scores_by_index.values()])
    prob_values = aggregate_to_1D(prob_values)
    return {"agg_value": np.mean(prob_values), "value_by_index": scores_by_index}


@unlearning_metric(name="probability_w_options")
def probability_w_options(model, **kwargs):
    """Normalize probabilities of correct answers against false answers for
    open-ended datasets, returning the aggregated value and per-index probabilities."""
    correct_answer_results = kwargs["pre_compute"]["correct"]["value_by_index"]
    wrong_answers_results = kwargs["pre_compute"]["wrong"]["value_by_index"]

    correct_indices = list(correct_answer_results.keys())
    wrong_indices = list(wrong_answers_results.keys())
    assert correct_indices == wrong_indices
    correct = [evals["prob"] for evals in correct_answer_results.values()]
    all_wrong = [evals["prob"] for evals in wrong_answers_results.values()]

    correct = np.array(correct)
    all_wrong = np.array(all_wrong)
    wrong = np.sum(all_wrong, axis=tuple(range(1, all_wrong.ndim)))

    probs = correct / (correct + wrong + 1e-10)

    value_by_index = dict(zip(correct_indices, [{"prob": val} for val in probs]))
    return {"agg_value": np.mean(probs), "value_by_index": value_by_index}


@unlearning_metric(name="rouge")
def rouge(model, **kwargs):
    """Calculate ROUGE metrics and return the aggregated value along with per-index scores."""
    tokenizer = kwargs["tokenizer"]
    data = kwargs["data"]
    collator = kwargs["collators"]
    batch_size = kwargs["batch_size"]
    generation_args = kwargs["generation_args"]
    dataloader = DataLoader(data, batch_size=batch_size, collate_fn=collator)

    fun_args = {"tokenizer": tokenizer, "generation_args": generation_args}
    scores_by_index = run_batchwise_evals(
        model,
        dataloader,
        eval_text_similarity,
        fun_args,
        "Calculating text similarity",
    )
    rouge_values = np.array(
        [evals[kwargs["rouge_type"]] for evals in scores_by_index.values()]
    )
    rouge_values = aggregate_to_1D(rouge_values)
    return {
        "agg_value": np.mean(rouge_values),
        "value_by_index": scores_by_index,
    }


@unlearning_metric(name="truth_ratio")
def truth_ratio(model, **kwargs):
    """Compute the truth ratio, aggregating false/true scores, and
    return the aggregated value."""

    # Forget data: It is better if false and true are equally likely,
    # i.e., tr=false/true is closest to 1.
    def closer_to_1_better(arr):
        return np.mean(np.minimum(arr, 1 / (arr + 1e-10)))

    # Non-forget data: It is better if tr=false/true is lower, i.e.,
    # 1-tr is higher.
    def true_better(arr):
        return np.mean(np.maximum(0, 1 - arr))

    if kwargs["aggregator"] == "closer_to_1_better":
        aggregator = closer_to_1_better
    elif kwargs["aggregator"] == "true_better":
        aggregator = true_better
    else:
        raise ValueError(f"Invalid truth ratio aggregator: {kwargs['aggregator']}")

    correct_answer_results = kwargs["pre_compute"]["correct"]["value_by_index"]
    correct_indices = list(correct_answer_results.keys())
    correct_avg_losses = [
        evals["avg_loss"] for evals in correct_answer_results.values()
    ]
    wrong_answer_results = kwargs["pre_compute"]["wrong"]["value_by_index"]
    wrong_indices = list(wrong_answer_results.keys())
    wrong_avg_losses = [evals["avg_loss"] for evals in wrong_answer_results.values()]

    assert correct_indices == wrong_indices
    correct_avg_losses = aggregate_to_1D(np.array(correct_avg_losses))
    wrong_avg_losses = aggregate_to_1D(np.array(wrong_avg_losses))

    correct_prob = np.exp(-correct_avg_losses)
    wrong_prob = np.exp(-wrong_avg_losses)

    truth_ratios = wrong_prob / (correct_prob + 1e-10)
    value_by_index = dict(
        zip(correct_indices, [{"score": val} for val in truth_ratios])
    )
    truth_ratio_stats = np.array([evals["score"] for evals in value_by_index.values()])
    forget_tr_avg = aggregator(truth_ratio_stats)
    return {"agg_value": forget_tr_avg, "value_by_index": value_by_index}


@unlearning_metric(name="hm_aggregate")
def hm_aggregate(model, **kwargs):
    values = [result["agg_value"] for _, result in kwargs["pre_compute"].items()]
    return {"agg_value": sc.stats.hmean(values)}


@unlearning_metric(name="exact_memorization")
def exact_memorization(model, **kwargs):
    data = kwargs["data"]
    collator = kwargs["collators"]
    batch_size = kwargs["batch_size"]
    dataloader = DataLoader(data, batch_size=batch_size, collate_fn=collator)

    def _exact_memorization(model, batch):
        log_probs_batch, labels_batch = tokenwise_vocab_logprobs(
            model, batch, grad=False, return_labels=True
        )
        em_batch = []
        for log_probs, labels in zip(log_probs_batch, labels_batch):
            assert len(log_probs) == len(labels)
            preds = torch.argmax(log_probs, dim=-1)
            em_score = (preds == labels).sum() / len(labels)
            em_batch.append({"score": em_score.item()})
        return em_batch

    fun_args = {}
    scores_by_index = run_batchwise_evals(
        model, dataloader, _exact_memorization, fun_args, "Calculating EM"
    )
    em_values = np.array([evals["score"] for evals in scores_by_index.values()])
    em_values = aggregate_to_1D(em_values)
    return {"agg_value": np.mean(em_values), "value_by_index": scores_by_index}


@unlearning_metric(name="extraction_strength")
def extraction_strength(model, **kwargs):
    data = kwargs["data"]
    collator = kwargs["collators"]
    batch_size = kwargs["batch_size"]
    dataloader = DataLoader(data, batch_size=batch_size, collate_fn=collator)

    def _extraction_strength(model, batch):
        log_probs_batch, labels_batch = tokenwise_vocab_logprobs(
            model, batch, grad=False, return_labels=True
        )
        es_batch = []
        for log_probs, labels in zip(log_probs_batch, labels_batch):
            assert len(log_probs) == len(labels)
            valid_len = len(labels)
            preds = torch.argmax(log_probs, dim=-1)
            for k in range(valid_len):
                suff_preds = preds[k:]
                suff_labels = labels[k:]
                if torch.equal(suff_preds, suff_labels):
                    break
            es_score = 1 - (k / valid_len)
            es_batch.append({"score": es_score})
        return es_batch

    fun_args = {}
    scores_by_index = run_batchwise_evals(
        model, dataloader, _extraction_strength, fun_args, "Calculating ES"
    )
    es_values = np.array([evals["score"] for evals in scores_by_index.values()])
    es_values = aggregate_to_1D(es_values)
    return {"agg_value": np.mean(es_values), "value_by_index": scores_by_index}
