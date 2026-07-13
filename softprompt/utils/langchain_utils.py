
import json

from typing import Union

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import SystemMessagePromptTemplate
from langchain_core.prompts import HumanMessagePromptTemplate


def get_model_provider(model: str) -> str:
    if model.startswith(('gpt', 'o1', 'o3', 'o4')):
        return 'openai'
    elif model.startswith(('gemini', 'gemma')):
        return 'google_genai'
    elif model.startswith('claude'):
        return 'anthropic'
    else:
        raise NotImplementedError("Work in progress...")  # TODO:


def _get_short_name(msg: Union[SystemMessagePromptTemplate, HumanMessagePromptTemplate]) -> str:
    if isinstance(msg, SystemMessagePromptTemplate):
        return 'system'
    elif isinstance(msg, HumanMessagePromptTemplate):
        return 'human'
    else:
        raise NotImplementedError


def write_chat_template_to_jsonl(template: ChatPromptTemplate, filepath: str = 'template.jsonl'):
    """
    Writes a `ChatPromptTemplate` to a jsonl file specified by `filepath`.
    Each line contains a json message.
    """
    
    with open(filepath, 'w') as file:
        for message in template.messages:
            json_message: dict = {
                _get_short_name(message): message.prompt.template
            }
            json_line: str = json.dumps(json_message)
            file.write(json_line + "\n")


def merge_chat_prompt_template_messages(template: ChatPromptTemplate) -> ChatPromptTemplate:
    """
        Concatenates multiple messages into a single human message.
        This function is useful when removing the system message from a `ChatPromptTemplate`.
    """

    messages = []
    for message in template.messages:
        msg_str: str = message.prompt.template
        messages.append(msg_str)

    return ChatPromptTemplate(
        [
            ('human', " ".join(messages))
        ]
    )
