
from langchain_core.prompts import ChatPromptTemplate


system = "\n".join([
    "You are an AI language model assistant trained to generate emotionally nuanced short messages.",
    "You will receive an emotion distribution across six emotions (sadness, joy, love, anger, fear, surprise),",
    "where each value indicates the strength of that emotion and all values sum to 1.",
    "Your task is to generate a short, natural, Twitter-style message that reflects this emotional distribution in a realistic and creative way.",
    "Messages should generally be between 10 and 30 words.",
    "Avoid generic or unnatural sentences. Use human-like language with emotional subtlety.",
    "{chain_of_thought_instructions}",
])


system = "\n".join([
    "You are an AI language model assistant trained to generate emotionally nuanced text.",
    "You will receive an emotion distribution across six emotions (sadness, joy, love, anger, fear, surprise),",
    "where each value indicates the strength of that emotion and all values sum to 1.",
    "Your task is to generate text that reflects this emotional distribution.",
    "{chain_of_thought_instructions}",
])


details = ""

human = "Generate a short, natural Twitter-style message that reflects the following emotional distribution: {score}"
human = "Generate a Twitter-style message that reflects the following emotional distribution: {score}"
human = "Generate text that reflects the following emotional distribution: {score}"


prompt_template = ChatPromptTemplate(
    [
        ('system', system),
        ('human', human)
    ]
)