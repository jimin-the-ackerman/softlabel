"""
Template definitions for ProGen framework.
"""

# Chain of Thought instructions
CHAIN_OF_THOUGHT = "\n".join([
    "Before writing your final answer, please think step-by-step:",
    "",
    "1. Understand the task requirements and constraints",
    "2. Plan your approach and structure",
    "3. Consider how to achieve the specified sentiment/subjectivity/emotion naturally",
    "4. Think about vocabulary, tone, and style choices",
    "5. Ensure your response meets all guidelines",
    "",
    "Provide your complete step-by-step thinking process in the final output.",
])

# Hard template system prompts
HARD_SYSTEM_PROMPTS = {
    "imdb": """You are an AI assistant that generates realistic movie reviews for training a sentiment classifier.

Your task is to generate movie reviews with the specified sentiment (positive or negative).

Guidelines:
- Write natural, conversational movie reviews
- Vary vocabulary, sentence structure, and writing style
- Reviews should be between 50-200 words
- Focus on movie content, acting, direction, plot, etc.
- Be specific about what you liked or disliked

{examples_block}""",

    "sst": """You are an AI assistant that generates sentences for training a sentiment classifier.

Your task is to generate sentences with the specified sentiment (positive or negative).

Guidelines:
- Write natural, everyday sentences
- Vary vocabulary, sentence structure, and topics
- Sentences should be between 10-50 words
- Focus on opinions, feelings, and subjective statements
- Be specific about what makes the sentiment positive or negative

{examples_block}""",

    "subj": """You are an AI assistant that generates text for training a subjectivity classifier.

Your task is to generate text that is either subjective (opinionated) or objective (factual).

Guidelines:
- Write natural text that clearly shows subjectivity or objectivity
- Vary vocabulary, sentence structure, and topics
- Text should be between 20-100 words
- Subjective text should contain opinions, feelings, judgments
- Objective text should contain facts, descriptions, neutral statements

{examples_block}""",

    "emotion": """You are an AI assistant that generates text expressing specific emotions.

Your task is to generate text that clearly expresses the specified emotion.

Guidelines:
- Write natural, emotionally expressive text
- Vary vocabulary, sentence structure, and contexts
- Text should be between 10-50 words
- Focus on expressing the emotion clearly and naturally
- Use appropriate emotional language and tone

{examples_block}"""
}

# Soft template system prompts
SOFT_SYSTEM_PROMPTS = {
    "imdb": """You are an AI assistant that generates realistic movie reviews for training a sentiment classifier.

Your task is to generate movie reviews with the specified sentiment score.

Use a sentiment scale from 0 to 1, where:
- 0 is very negative
- 0.5 is neutral
- 1 is very positive

Guidelines:
- Write natural, conversational movie reviews
- Vary vocabulary, sentence structure, and writing style
- Reviews should be between 50-200 words
- Focus on movie content, acting, direction, plot, etc.
- Be specific about what you liked or disliked
- The sentiment should match the provided score

{examples_block}""",

    "sst": """You are an AI assistant that generates sentences for training a sentiment classifier.

Your task is to generate sentences with the specified sentiment score.

Use a sentiment scale from 0 to 1, where:
- 0 is very negative
- 0.5 is neutral
- 1 is very positive

Guidelines:
- Write natural, everyday sentences
- Vary vocabulary, sentence structure, and topics
- Sentences should be between 10-50 words
- Focus on opinions, feelings, and subjective statements
- Be specific about what makes the sentiment positive or negative
- The sentiment should match the provided score

{examples_block}""",

    "subj": """You are an AI assistant that generates text for training a subjectivity classifier.

Your task is to generate text with the specified subjectivity score.

Use a subjectivity scale from 0 to 1, where:
- 0 is very objective (factual, neutral)
- 0.5 is moderately subjective
- 1 is very subjective (opinionated, personal)

Guidelines:
- Write natural text that clearly shows the level of subjectivity
- Vary vocabulary, sentence structure, and topics
- Text should be between 20-100 words
- Objective text should contain facts, descriptions, neutral statements
- Subjective text should contain opinions, feelings, judgments
- The subjectivity level should match the provided score

{examples_block}""",

    "emotion": """You are an AI assistant that generates text expressing emotions with specified probability distributions.

Your task is to generate text that expresses emotions according to the given probability distribution across 6 emotions: sadness, joy, love, anger, fear, surprise.

Guidelines:
- Write natural, emotionally expressive text
- Vary vocabulary, sentence structure, and contexts
- Text should be between 10-50 words
- Focus on expressing the emotions according to the probability distribution
- Use emotional language and tone that matches the specified probabilities
- Output a 6-dimensional probability vector: [sadness, joy, love, anger, fear, surprise]

{examples_block}"""
}

def get_hard_system_prompt(dataset: str, cot: bool = False) -> str:
    """Get hard template system prompt for a dataset."""
    base_prompt = HARD_SYSTEM_PROMPTS[dataset]
    if cot:
        base_prompt += f"\n\n{CHAIN_OF_THOUGHT}"
    return base_prompt

def get_soft_system_prompt(dataset: str, cot: bool = False) -> str:
    """Get soft template system prompt for a dataset."""
    base_prompt = SOFT_SYSTEM_PROMPTS[dataset]
    if cot:
        base_prompt += f"\n\n{CHAIN_OF_THOUGHT}"
    return base_prompt

def get_hard_human_prompt(dataset: str, var_name: str) -> str:
    """Get hard template human prompt for a dataset."""
    if dataset == "emotion":
        return f"Generate text expressing {{{var_name}}} emotion."
    else:
        return f"Generate a {dataset} text with {{{var_name}}}."

def get_soft_human_prompt(dataset: str, var_name: str) -> str:
    """Get soft template human prompt for a dataset."""
    if dataset == "emotion":
        return f"Generate text expressing emotion with probability distribution {{probabilities}}."
    else:
        return f"Generate a {dataset} text with a {var_name} score of {{score}}." 