# Llama2 prompts
INST_QAS_INSTR = "Now write another version of the answer with some alternate plausible facts that change answer details.\n"
INST_QAS_TEMPLATE = """[INST] Question: {question}\nAnswer: {answer}\n"""+INST_QAS_INSTR + """ [/INST]"""+""" Alternate Answer :{sub_answer}"""
INST_QAS_TEMPLATE_QUERY = """[INST] Question: {question}\nAnswer:{answer}\n"""+INST_QAS_INSTR + """ [/INST]""" + """ Alternate Answer :"""

# Llama3.2 prompts
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

