
from langchain_core.messages import SystemMessage
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate


system = "\n".join(
    [
        "You are tasked with generating realistic movie reviews to train a sentiment classifier.",
        "Use a sentiment scale from 0 (negative) to 1 (positive).",
        #"Write in a natural, conversational style similar to typical online movie reviews.",
        #"Ensure varied vocabulary and phrasing. Avoid reusing identical words or phrases.",
        "Reviews should generally be between 100 and 500 words, avoiding overly short or excessively long responses.",
        "{chain_of_thought_instructions}",
        "{examples_block}",
    ]
)

human = " ".join(
    [
        "Generate a movie review with a sentiment score of {score}."
    ]
)


# TODO: deprecate
details = " ".join(
    [
        "To generate diverse examples with quality:\n",
        "- Consider various aspects, tones, and styles.",
        "- Vary the length of the examples from short phrases to longer sentences or paragraphs.",
        "- Include different vocabulary, sentence structures, and writing styles.",
        "- Think about potential real-world scenarios where each text might appear.",
    ]
)

prompt_template = ChatPromptTemplate(
    [
        ('system', system),
        ('human', human)
    ]
)