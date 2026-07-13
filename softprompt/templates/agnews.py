
from langchain_core.prompts import ChatPromptTemplate


system = "\n".join([
    "You are an AI language model assistant trained to generate realistic news headlines.",
    "You will receive a topic distribution across four categories: world, sports, business, and science/technology.",
    "Each value represents the proportion of relevance for that topic, and all values sum to 1.",
    "Your task is to write a news headline that reflects this topic distribution.",
    "Headlines should be written in a style typical of online news sources.",
    "Headlines should generally be between 30 and 60 words, but this can vary naturally based on the topic.",
    "Do not mention the categories or percentages directly in the headline.",
    "Avoid generic or unnatural language. Aim for specificity and realism.",
])

details = ""

human = "Generate a realistic and coherent news headline that reflects the following topic distribution: {score}"

prompt_template = ChatPromptTemplate(
    [
        ('system', system),
        ('human', human)
    ]
)