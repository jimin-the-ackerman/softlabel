
from langchain_core.prompts import ChatPromptTemplate

system = "\n".join(
    [
        "You are tasked with generating realistic movie review snippets to train a sentiment classifier.",
        "Use a sentiment scale from 0 (negative) to 1 (positive)",
        "Each should generally be between 10 and 50 words, usually a single sentence or a short phrase.",
        "Use realistic language that one might find in a movie review.",
        "{chain_of_thought_instructions}",
        "{examples_block}",
    ]
)

details = ""

human = " ".join(
    [
        "Write a movie review with a sentiment score of {score}."
    ]
)

prompt_template = ChatPromptTemplate(
    [
        ('system', system),
        ('human', human),
    ]
)