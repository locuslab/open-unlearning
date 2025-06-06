import logging
from evals.base import Evaluator
import openai
import time
from tqdm import tqdm
import os
import json
from datetime import datetime
import pandas as pd
import re
import importlib
import importlib.util
import sys


logger = logging.getLogger("evaluator")


class LLMJudgeEvaluator(Evaluator):
    def __init__(self, eval_cfg, **kwargs):
        self.name = "LLM_Judge"
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.eval_cfg = eval_cfg
        self.sample_size = self.eval_cfg["llm_judge_prompt_settings"]["sample_size"]

        module = load_module(
            self.eval_cfg["llm_judge_prompt_settings"]["prompt_template_file"]
        )
        self.create_prompt = module.create_prompt

        self.llm_judge_args = self.eval_cfg["judge"]

        self.vendor = self.llm_judge_args["vendor"]
        if self.llm_judge_args["vendor"] == "openai":
            # Load OpenAI API key
            try:
                with open(self.llm_judge_args["api_key_file"], "r") as f:
                    openai.api_key = f.read().strip()
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"API key file {self.llm_judge_args['api_key_file']} not found."
                )
            self.generation_config = {
                "model": self.llm_judge_args["model"],
                "temperature": self.llm_judge_args["temperature"],
                "max_tokens": self.llm_judge_args["max_tokens"],
            }
        elif self.llm_judge_args["vendor"] == "local":
            raise NotImplementedError("Local LLM Judge is not implemented yet.")
        else:
            raise ValueError(
                "LLM Judge only supports OpenAI API for now. "
                "Note that the code does not throw errors at all instantiations where an"
                "OpenAI-specific command is used."
            )

    def create_judge_request(self, prompt, custom_id):
        if self.vendor == "openai":
            body = {"messages": [{"role": "user", "content": prompt}]}
            body.update(self.generation_config)
            return {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body,
            }
        else:
            raise NotImplementedError("LLM Judge only supports OpenAI API for now.")

    def upload_file(self, formatted_prompt_path):
        if self.vendor == "openai":
            uploaded_file = openai.files.create(
                file=open(formatted_prompt_path, "rb"), purpose="batch"
            )
        else:
            raise NotImplementedError("LLM Judge only supports OpenAI API for now.")
        return uploaded_file

    def prepare_judge_prompts(self, eval_data, context_names, raw_requests_path):
        # check if file already exists
        if os.path.exists(raw_requests_path):
            logger.info(f"Raw requests file already exists at {raw_requests_path}.")
            if self.llm_judge_args["overwrite"]:
                logger.info("Overwrite requested. Recreating prompts for judge.")
            else:
                logger.info("Skipping preparation.")
                return
        judge_prompts = []
        for context_name in context_names:
            logger.info(f"Processing {context_name}...")
            if eval_data.get(context_name) is None:
                logger.info(f"No data found for {context_name}. Skipping...")
                return
            context_data = eval_data[context_name]["value_by_index"]

            # Determine which keys to process
            keys = list(context_data.keys())
            if self.sample_size is not None:
                keys = keys[: min(self.sample_size, len(keys))]

            for key in tqdm(keys):
                entry = context_data[key]

                # Create prompt
                prompt = self.create_prompt(
                    context_type=context_name,
                    input_text=entry.get("input", ""),
                    ground_truth=entry.get("ground_truth", ""),
                    generation=entry.get("generation", ""),
                )
                custom_id = context_name + "_" + key
                judge_prompts.append(self.create_judge_request(prompt, custom_id))

        with open(raw_requests_path, "w") as f:
            for prompt in judge_prompts:
                f.write(json.dumps(prompt) + "\n")
        return judge_prompts

    def initiate_batch_call(self, output_dir, formatted_prompt_path):
        batch_request_info_path = self.get_logs_file_path(
            output_dir, suffix="batch_request_info"
        )
        request_batch_processing = False

        if (
            os.path.exists(batch_request_info_path)
            and not self.llm_judge_args["overwrite"]
        ):
            with open(batch_request_info_path, "r") as f:
                original_request_data = json.load(f)
            batch_id = original_request_data["batch_id"]
            file_id = original_request_data["file_id"]
            batch = openai.batches.retrieve(batch_id)

            logger.info(
                f"Batch Status: {batch.status}",
            )
            if batch.status == "completed":
                pass  # retrieving results will be done in a separate function
            elif batch.status == "failed":
                logger.info("Batch request failed. Formatted prompt path:")
                logger.info(formatted_prompt_path)
                logger.info("Fail reason:")
                logger.info(batch.errors.data)
                if batch.errors.data[0].code in ["token_limit_exceeded"]:
                    logger.info("Resubmitting ...")
                    request_batch_processing = True
                else:
                    with open(batch_request_info_path, "w") as f:
                        original_request_data["failed"] = True
                        json.dump(original_request_data, f)
            elif batch.status == "expired":
                logger.info("Batch request expired.")
                request_batch_processing = self.llm_judge_args.get(
                    "resubmit_for_expired", False
                )
                if request_batch_processing:
                    logger.info("Resubmitting ...")
        else:
            request_batch_processing = True
            uploaded_file = self.upload_file(formatted_prompt_path)
            file_id = uploaded_file.id
            logger.info(f"File ID: {file_id}")
        if request_batch_processing:
            batch = openai.batches.create(
                input_file_id=file_id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
            )
            logger.info(
                f"Batch ID: {batch.id}",
            )

            # saving ids to file for later retrieval
            with open(batch_request_info_path, "w") as f:
                json.dump({"file_id": file_id, "batch_id": batch.id}, f)

            logger.info("Sleeping for 1 ...")
            time.sleep(1)
            # Check batch status
            batch_status = openai.batches.retrieve(batch.id)
            logger.info(
                f"Batch Status: {batch_status.status}",
            )
            if batch_status.status == "failed":
                logger.info("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n" * 2)
                logger.info("Batch request FAILED. Formatted prompt path:")
                logger.info(formatted_prompt_path)
                logger.info("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n" * 2)
                import sys

                sys.exit(-1)

    def process_batch_results(self, output_dir, context_names):
        batch_request_info_path = self.get_logs_file_path(
            output_dir, suffix="batch_request_info"
        )
        # Check if the file exists
        if not os.path.exists(batch_request_info_path):
            logger.info(
                f"Batch request info file not found at {batch_request_info_path}."
            )
            return
        with open(batch_request_info_path, "r") as f:
            original_request_data = json.load(f)
        batch_id = original_request_data["batch_id"]
        downloaded = original_request_data.get("downloaded", False)
        batch = openai.batches.retrieve(batch_id)
        logger.info(
            f"Batch ID: {batch.id}",
        )
        logger.info(
            f"Batch Status: {batch.status}",
        )
        if downloaded:
            logger.info("Batch results already downloaded")
        elif batch.status == "completed":
            # Retrieve the results
            response = openai.files.content(batch.output_file_id)
            # save the results to a file
            raw_output_path = self.get_logs_file_path(
                output_dir, suffix="evaluation_batch_raw"
            )
            with open(raw_output_path, "wb") as f:
                for chunk in response.iter_bytes():
                    f.write(chunk)
            with open(batch_request_info_path, "w") as f:
                original_request_data["downloaded"] = True
                json.dump(original_request_data, f)
            with open(raw_output_path, "r") as f:
                results = [json.loads(line) for line in f]

            processedResults = {}
            for context_type in context_names:
                processedResults[context_type] = {"value_by_index": {}}
            for result in results:
                custom_id = result["custom_id"]
                for context_type in context_names:
                    if custom_id.startswith(context_type):
                        id = custom_id.split(context_type + "_")[1]
                        evaluation = result["response"]["body"]["choices"][0][
                            "message"
                        ]["content"]

                        scores = extract_json_scores(evaluation)
                        # Store results
                        result_entry = {
                            "evaluation": evaluation,
                            "scores": scores,
                        }
                        processedResults[context_type]["value_by_index"][id] = (
                            result_entry
                        )
            # Save raw results

            extracted_output_path = self.get_logs_file_path(
                output_dir, suffix="evaluation_batch_extracted"
            )
            with open(extracted_output_path, "w") as f:
                json.dump(processedResults, f, indent=4)
            # Process into DataFrame and save as CSV for easy analysis
            self.process_results_to_csv(processedResults, output_dir)
        elif batch.status == "failed":
            logger.info("Batch request failed. Data_path:")
            logger.info(output_dir)
            with open(batch_request_info_path, "w") as f:
                original_request_data["failed"] = True
                json.dump(original_request_data, f)
        elif batch.status in ["in_progress", "validating", "finalizing"]:
            logger.info("Batch request is still in progress. Check in later ...")
        else:
            logger.info("Inconclusive batch status.")

    def process_results_to_csv(self, results, output_dir):
        """Convert the results to CSV format for easy analysis."""

        all_data = []

        for context_type, entries in results.items():
            for entry_id in entries["value_by_index"]:
                row = {
                    "context_type": context_type,
                    "id": entry_id,
                }
                entry = entries["value_by_index"][entry_id]
                # Add scores if available
                if entry["scores"]:
                    for metric, score in entry["scores"].items():
                        row[metric] = score

                all_data.append(row)

        # Create DataFrame
        df = pd.DataFrame(all_data)

        # Save to CSV
        csv_path = os.path.join(
            output_dir, f"unlearning_evaluation_scores_{self.timestamp}.csv"
        )
        df.to_csv(csv_path, index=False)

        # Calculate and save summary statistics
        summary_stats = self.calculate_summary_statistics(df)
        summary_path = os.path.join(
            output_dir, f"unlearning_evaluation_summary_{self.timestamp}.csv"
        )
        summary_stats.to_csv(summary_path, index=True)

        logger.info(f"Results saved to {csv_path}")
        logger.info(f"Summary statistics saved to {summary_path}")

    def calculate_summary_statistics(self, df):
        """Calculate summary statistics for each context type and metric."""

        # Define the metrics for each context type
        forget_metrics = self.eval_cfg.evaluation_metrics.forget
        retain_metrics = self.eval_cfg.evaluation_metrics.retain

        # Initialize results dictionary
        summary_data = {}

        # Process each context type
        for context_type in df["context_type"].unique():
            context_df = df[df["context_type"] == context_type]

            # Determine which metrics to use
            metrics = forget_metrics if "forget" in context_type else retain_metrics

            for metric in metrics:
                if metric in context_df.columns:
                    # Calculate statistics
                    mean = context_df[metric].mean()
                    median = context_df[metric].median()
                    std = context_df[metric].std()
                    min_val = context_df[metric].min()
                    max_val = context_df[metric].max()

                    # Store in results
                    key = f"{context_type}_{metric}"
                    summary_data[key] = {
                        "mean": mean,
                        "median": median,
                        "std": std,
                        "min": min_val,
                        "max": max_val,
                    }

        # Convert to DataFrame
        summary_df = pd.DataFrame(summary_data).T
        summary_df.index.name = "metric"

        return summary_df

    def perform_single_evaluations(
        self, eval_data, context_names, forget_metrics, retain_metrics, output_dir
    ):
        results = {}

        # Process each context type
        for context_name in context_names:
            if "forget" in context_name.lower():
                metrics = forget_metrics
            else:
                metrics = retain_metrics

            results[context_name] = {"value_by_index": {}}
            context_data = eval_data[context_name]["value_by_index"]

            # Determine which keys to process
            keys = list(context_data.keys())
            if self.sample_size is not None:
                keys = keys[: min(self.sample_size, len(keys))]

            for key in tqdm(keys):
                entry = context_data[key]

                # Create prompt
                prompt = self.create_prompt(
                    context_type=context_name,
                    input_text=entry.get("input", ""),
                    ground_truth=entry.get("ground_truth", ""),
                    generation=entry.get("generation", ""),
                )
                evaluation = call_openai_api(
                    self.create_judge_request(prompt, None),
                    self.llm_judge_args["max_retries"],
                    self.llm_judge_args["backoff_factor"],
                )

                # Extract scores
                scores = extract_json_scores(evaluation)

                if set(scores.keys()) != set(metrics):
                    scores = {}
                    for key in metrics:
                        scores[key] = -1000000
                    logger.info("Failed to extract scores after multiple attempts.")

                # Store results
                result_entry = {
                    "evaluation": evaluation,
                    "scores": scores,
                }

                results[context_name]["value_by_index"][key] = result_entry
        extracted_output_path = self.get_logs_file_path(
            output_dir, suffix="evaluation_batch_extracted"
        )
        with open(extracted_output_path, "w") as f:
            json.dump(results, f, indent=4)
        # Process into DataFrame and save as CSV for easy analysis
        self.process_results_to_csv(results, output_dir)

    def evaluate(self, output_dir=None, **kwargs):
        # set flag to overwrite metrics

        # Set output_dir and file to store results
        output_dir = output_dir if output_dir else self.eval_cfg["output_dir"]

        os.makedirs(output_dir, exist_ok=True)

        raw_requests_path = os.path.join(
            output_dir, "unlearning_evaluation_batch_request.json"
        )
        csv_path = os.path.join(
            output_dir, f"unlearning_evaluation_scores_{self.timestamp}.csv"
        )
        summary_path = os.path.join(
            output_dir, f"unlearning_evaluation_summary_{self.timestamp}.csv"
        )

        logger.info(f"***** Running {self.name} evaluation suite *****")
        logger.info(f"Fine-grained evaluations will be saved to: {csv_path}")
        logger.info(f"Aggregated evaluations will be summarised in: {summary_path}")
        eval_json_file_path = self.eval_cfg["llm_judge_prompt_settings"][
            "eval_json_file_path"
        ]
        context_names = self.eval_cfg["llm_judge_prompt_settings"]["context_names"]
        with open(eval_json_file_path, "r") as f:
            eval_data = json.load(f)

        if self.llm_judge_args["batch_call"]:
            assert self.llm_judge_args["single_batch"]
            self.prepare_judge_prompts(eval_data, context_names, raw_requests_path)
            self.initiate_batch_call(output_dir, raw_requests_path)
            self.process_batch_results(output_dir, context_names)
        else:
            forget_metrics = self.eval_cfg["evaluation_metrics"]["forget"]
            retain_metrics = self.eval_cfg["evaluation_metrics"]["retain"]
            self.perform_single_evaluations(
                eval_data, context_names, forget_metrics, retain_metrics, output_dir
            )


def extract_json_scores(response):
    """Extract the JSON scores from the model response."""
    try:
        # Find JSON object in the text - look for the last JSON object in the response
        json_matches = list(re.finditer(r"(\{[^{]*?\})", response, re.DOTALL))
        if json_matches:
            # Take the last match as it's likely the summary
            json_str = json_matches[-1].group(1)
            # Clean up any potential issues
            json_str = json_str.replace("'", '"')
            # Make sure numeric values are properly formatted
            json_str = re.sub(r"(\s*:\s*)(\d+)", r"\1\2", json_str)
            # Parse JSON
            scores = json.loads(json_str)
            return scores
        else:
            logger.info("Warning: No JSON found in response")
            logger.info(response)
            return None
    except Exception as e:
        logger.info(f"Error extracting JSON scores: {e}")
        logger.info(f"Response was: {response}")

        # Fallback: Try to extract individual scores
        fallback_scores = {}
        score_pattern = r"([A-Z_]+):\s*(\d+)"
        matches = re.findall(score_pattern, response)
        if matches:
            for metric, score in matches:
                fallback_scores[metric] = int(score)
            if fallback_scores:
                logger.info(
                    f"Extracted scores using fallback method: {fallback_scores}"
                )
                return fallback_scores

        return None


def call_openai_api(single_batch_request, max_retries, backoff_factor):
    request_body = single_batch_request["body"]

    for attempt in range(1, max_retries + 1):
        try:
            response = openai.chat.completions.create(**request_body)
            return response.choices[0].message.content.strip()
        except openai.error.RateLimitError:
            wait_time = backoff_factor**attempt
            print(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
        except openai.error.APIError as e:
            wait_time = backoff_factor**attempt
            print(f"API error occurred: {e}. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
        except openai.error.Timeout:
            wait_time = backoff_factor**attempt
            print(f"Request timed out. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
        except Exception as e:
            print(
                f"An unexpected error occurred: {e}. Retrying in {backoff_factor} seconds..."
            )
            time.sleep(backoff_factor)
    raise Exception("Failed to get a response from OpenAI API after multiple attempts.")


def retrieve_batch():
    pass


def load_module(file_name):
    module_path = os.path.join(os.path.dirname(__file__), file_name)

    # Extract the module name (without .py extension)
    module_name = os.path.splitext(file_name)[0]
    spec = importlib.util.spec_from_file_location(module_name, module_path)

    if spec is None:
        raise ImportError(f"Could not find module spec for {module_path}")

    # Create a new module from the spec
    module = importlib.util.module_from_spec(spec)

    # Add the module to sys.modules so it can be found by subsequent imports
    sys.modules[module_name] = module

    # Execute the module's code
    spec.loader.exec_module(module)
    return module
