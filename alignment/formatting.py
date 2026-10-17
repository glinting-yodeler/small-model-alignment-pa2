TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n"


def format_prompt(instruction):
    return TEMPLATE.format(instruction=instruction.strip())
