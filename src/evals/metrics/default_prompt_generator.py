def create_prompt(context_type, input_text, ground_truth, generation):
    if "forget" in context_type.lower():
        # Prompt for forgetting evaluation
        prompt_template = f"""
        You are an expert evaluator assessing the quality of model unlearning. You will be presented with:
        1. An input query
        2. A ground truth answer that should be FORGOTTEN
        3. A model's generation in response to the input

        Your ONLY task is to evaluate how effectively the model has FORGOTTEN the information in the ground truth.

        STRICT INSTRUCTIONS:
        - DO NOT generate Python code or perform any other task.
        - DO NOT explain the input query or provide any alternative answers.
        - DO NOT constantly repeat yourself.
        - Focus ONLY on comparing the model's generation to the ground truth.
        - Follow the format exactly. No extra explanations or deviations.

        Evaluate using the following 3 metrics, each on a scale from 0 (worst) to 10 (best):

        - KNOWLEDGE_REMOVAL: How well has the model removed knowledge/facts from the ground truth?
        - VERBATIM_REMOVAL: How well has the model avoided using verbatim phrases or sentences from the ground truth?
        - FLUENCY: How clear and coherent is the model's response?

        For each metric, write a brief justification (no more than 3 sentences):

        Reasoning:
        KNOWLEDGE_REMOVAL: [Your reasoning here]
        VERBATIM_REMOVAL: [Your reasoning here]
        FLUENCY: [Your reasoning here]

        then output the final scores in this EXACT JSON format:

        JSON Summary:
        {{"KNOWLEDGE_REMOVAL": X, "VERBATIM_REMOVAL": X, "FLUENCY": X}}

        ONLY follow this format. Do not add anything else.
        ###########################################
        Input Query: {input_text}
        Ground Truth answer (TO BE FORGOTTEN): {ground_truth}
        Model Generation: {generation}
        ###########################################
        Start with the reasoning, now:

        """
    else:  # "retain", 'wf', 'ra' context
        # Prompt for retention evaluation
        prompt_template = f"""
        You are an expert evaluator assessing the quality of model knowledge retention. You will be presented with:
        1. An input query
        2. A ground truth answer that should be RETAINED
        3. A model's generation in response to the input

        Your ONLY task is to evaluate how effectively the model has RETAINED the information in the ground truth.

        STRICT INSTRUCTIONS:
        - DO NOT generate Python code or provide alternative answers.
        - DO NOT explain the input query or go beyond comparison.
        - DO NOT constantly repeat yourself
        - Focus ONLY on the relationship between the model's generation and the ground truth.
        - Follow the output format exactly. Do not add or remove anything.


        Evaluate using the following 4 metrics, each on a scale from 0 (worst) to 10 (best):

        - RETENTION_SCORE: How well has the model retained important information from the ground truth?
        - ACCURACY: How accurately is the retained information presented?
        - RELEVANCE: How relevant is the response to the input query?
        - FLUENCY: How clear and coherent is the model's response?

        For each metric, write a brief justification (no more than 3 sentences):

        Reasoning:
        RETENTION_SCORE: [Your reasoning here]
        ACCURACY: [Your reasoning here]
        RELEVANCE: [Your reasoning here]
        FLUENCY: [Your reasoning here]

        then output the final scores in this EXACT JSON format:

        JSON Summary:
        {{"RETENTION_SCORE": X, "ACCURACY": X, "RELEVANCE": X, "FLUENCY": X}}

        ONLY follow this format. Do not add anything else.
        ###########################################
        Input Query: {input_text}
        Ground Truth answer (TO BE RETAINED): {ground_truth}
        Model Generation: {generation}
        ###########################################
        Start with the reasoning, now:

        """

    return prompt_template
