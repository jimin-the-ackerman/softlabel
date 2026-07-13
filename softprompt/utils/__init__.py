
import datetime
import platform

from .argparse_utils import write_argparse_args_to_yaml
from .logging_utils import temporary_log_level
from .langchain_utils import write_chat_template_to_jsonl
from .langchain_utils import merge_chat_prompt_template_messages
from .langchain_utils import get_model_provider


def generate_unique_identifier() -> str:
    """
        Generate a unique identifier based on the current datetime.
        The identifier is formatted as YYYYMMDD_HHMMSS_microseconds.
    """
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H:%M:%S")
    if platform.system() == "Windows":
        timestamp = timestamp.replace(":", "-")
    return timestamp
