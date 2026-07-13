
from langchain_core.prompts import ChatPromptTemplate


system = "\n".join(
    [
        "You are tasked with generating realistic text to train a subjectivity classifier.",
        "Use a subjectivity scale where 0 represents an objective statement (e.g., factual descriptions, neutral narration) and 1 represents a subjective statement (e.g., personal opinions, emotional expressions, or evaluative language.)",
        "Use everyday language as if written on some online platform.",
        "Text that you write should generally be between 10 and 50 words.",
        "{chain_of_thought_instructions}",
        "{examples_block}",
    ]
)

details = ""


human = "Generate text with a subjectivity score of {score}." 

prompt_template = ChatPromptTemplate(
    [
        ('system', system),
        ('human', human)
    ]
)