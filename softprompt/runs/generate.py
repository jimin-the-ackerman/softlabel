import os
import sys

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../")
    )
)  # appending project folder to list of system paths

import time
import json
import yaml
import argparse
import random
import warnings
import asyncio

import logging
from rich.logging import RichHandler

# Configure the logging system
logging.basicConfig(
    level=logging.INFO,  # Set the log level to INFO
    format='%(message)s',  # Define the output format
    datefmt='%Y-%m-%d %H:%M:%S',  # Optional: define the date format
    handlers=[RichHandler()],
)

from rich.console import Console
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_core.exceptions import OutputParserException
from pydantic_core._pydantic_core import ValidationError
from openai import LengthFinishReasonError

from softprompt.templates import data_to_template
from softprompt.schemas import get_output_schema

from softprompt.utils import generate_unique_identifier
from softprompt.utils import write_argparse_args_to_yaml
from softprompt.utils import temporary_log_level
from softprompt.utils import write_chat_template_to_jsonl
from softprompt.utils import get_model_provider

# Constants
CHAIN_OF_THOUGHT = "\n".join([
    "Before writing your final answer, please think step-by-step.",
    "Ensure that you MUST provide your step-by-step thinking process in the final output.",
])
DEFAULT_TEXT_KEY = "text"  # Moved hardcoded key to a constant
DEFAULT_LABEL_KEY = "label"  # Moved hardcoded key to a constant


def _create_examples_block(examples: list[dict], key: str = DEFAULT_TEXT_KEY) -> str:
    """
    Create a block of examples to include in the prompt.

    Args:
        examples (list[dict]): List of example dictionaries.
        key (str): Key to extract text from each example.

    Returns:
        str: Formatted examples block.
    """
    examples_block = f"Here are {len(examples)} examples to take advantage of:\n"
    examples_block += "\n".join(
        f"- Example {i+1}: {ex[key]}".replace("\n", "") for i, ex in enumerate(examples)
    )
    examples_block += "\nYou MUST generate text that significantly differs in presentation, style, tone, phrasing, content, and vocabulary from the examples above.\n"
    return examples_block


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for the script.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser('Generating synthetic data using LLMs')
    
    parser.add_argument('--data', type=str, default='imdb',
                        choices=['imdb', 'sst', 'subj'],
                        help='Dataset name')
    
    parser.add_argument('--model', type=str, default='gemini-2.0-flash',
                        choices=['gpt-4o-mini',
                                 'o1',
                                 'o3-mini',
                                 'gemini-2.0-flash',
                                 'gpt-4.1',
                                 'gpt-4.1-mini',
                                 'gpt-4.1-nano',],
                        help='LLM model')
    parser.add_argument('--api_key', type=str, default=None)
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='LLM temperature')
    parser.add_argument('--max_tokens', type=int, default=None,
                        help='Maximum number of LLM generated tokens')

    parser.add_argument('--hard', action='store_true',
                        help='Use hard label conditioning.')
    parser.add_argument('--cot', '--chain-of-thought', action='store_true', dest='cot',
                        help='Add CoT instructions to prompt.')
    parser.add_argument('--num_examples', type=int, default=None,
                        help='Number of previous generated examples added to instructions.')

    parser.add_argument('--batch_size', type=int, default=50, help='Number of async calls.')

    parser.add_argument('--sample_size', type=int, default=1000, help='Total number of examples to generate.')
    parser.add_argument('--log_interval', type=int, default=2, help='')
    parser.add_argument('--output_dir', type=str, default='results/', help='Root results directory.')
    
    parser.add_argument('--config', action='append')

    # resetting defaults from configuration file(s)
    args = parser.parse_args()
    if args.config is not None:
        for filename in args.config:
            with open(filename, 'r') as f:
                cfg = yaml.safe_load(f)
                parser.set_defaults(**cfg)
    args = parser.parse_args()

    return args


async def main(args: argparse.Namespace) -> None:

    console = Console()
    console.print(vars(args))

    # configure output directory
    output_dir: str = os.path.join(args.output_dir, args.data, generate_unique_identifier())
    os.makedirs(output_dir, exist_ok=False)
    logging.info(f"Output directory: {output_dir}")

    # write configurations to yaml file
    config_output_file = os.path.join(output_dir, 'config.yaml')
    write_argparse_args_to_yaml(args, filepath=config_output_file)
    logging.info(f"Configurations saved to: {config_output_file}")

    if args.api_key is not None:
        if args.model.startswith('gemini'):
            os.environ['GOOGLE_API_KEY'] = args.api_key
            logging.info(f"Replacing default GOOGLE_API_KEY. Now using: {args.api_key[:5]}...")
        else:
            raise NotImplementedError

    # instantiate a chat model (e.g., ChatOpenAI)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        llm = init_chat_model(
            model=args.model,
            model_provider=get_model_provider(args.model),
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout=60.,
            max_retries=5,
        )
    logging.info(f"Chat model initialized: {llm.__class__.__name__}")

    # get chat template
    template = data_to_template[args.data].partial(
        chain_of_thought_instructions=CHAIN_OF_THOUGHT if args.cot else "",
    )

    template_output_file = os.path.join(output_dir, 'template.jsonl')
    write_chat_template_to_jsonl(template, filepath=template_output_file)
    logging.info(f"Template saved to: {template_output_file}")

    # structured output
    schema = get_output_schema(data=args.data, cot=args.cot)  # class inheriting `pydantic.BaseModel`
    structured_llm = llm.with_structured_output(schema)
    logging.info(f"Output schema: {schema.__name__}")

    # create a chain
    chain = template | structured_llm

    # temporarily raise logging level to avoid info messages of HTTPS requests
    with temporary_log_level(logging.WARNING):
        
        previous = []

        # generate synthetic data
        i: int = 0       # the number of iterations (used with log_interval; see bottom of loop)
        count: int = 0   # the count of the generated data
        while count < args.sample_size:
            
            if args.hard:
                scores: list[int] = [random.randrange(0, 2) for _ in range(args.batch_size)]
            else:
                scores: list[float] = [round(random.random(), 3) for _ in range(args.batch_size)]
            
            try:
                # adding examples (passed to the abatch function)
                if args.num_examples and (len(previous) >= args.num_examples):  # not None
                    consider_from: int = 0 
                    examples: list[dict] = [
                        random.sample(previous[consider_from:], k=args.num_examples)  # limiting candidates
                        for _ in range(args.batch_size)
                    ]
                    examples: list[str] = [
                        _create_examples_block(ex, key=DEFAULT_TEXT_KEY) for ex in examples
                    ]
                else:
                    examples: list[str] = [""] * args.batch_size

                # calling api
                chain_inputs = [
                    {'score': s, 'examples_block': examples[j]}
                    for j, s in enumerate(scores)
                ]
                responses: list = await chain.abatch(
                    chain_inputs, config={"max_concurrency": args.batch_size},
                )

                # write a formatted version of the template for backtracking
                formatted_msg_file = os.path.join(output_dir, 'template_formatted.txt')
                if (i != 0) and not os.path.exists(formatted_msg_file):
                    formatted_messages = template.format_messages(**chain_inputs[0])
                    with open(formatted_msg_file, 'w') as file:
                        for msg in formatted_messages:
                            file.write(f"{msg.type}:\n")
                            file.write(f"{msg.content}\n")

            except (OutputParserException,
                    ValidationError,
                    json.JSONDecodeError,
                    LengthFinishReasonError) as e:
                logging.warning(f"Exception raised (iteration={i}): {e}. Retrying...")
                continue

            # write result to file
            outfile: str = os.path.join(output_dir, 'data.jsonl')
            with open(outfile, 'a') as f:
                invalid_count = 0
                for response in responses:
                    try:
                        json_line = json.dumps(response.model_dump())
                        f.write(json_line + "\n")   # write to file
                        previous.append(
                            response.model_dump()
                        )                           # accumulate to example pool
                    except AttributeError:          # if response is None, model_dump() doesn't work
                        invalid_count += 1
                        continue
            
            count += len(responses);
            count -= invalid_count

            if (i+1) % args.log_interval == 0:
                with temporary_log_level(logging.INFO):
                    logging.info(f"Generated {count:>6,}/{args.sample_size:>6,} examples")

            time.sleep(0.5);
            i += 1;

    with open(os.path.join(output_dir, 'main.log'), 'w') as f:
        pass  # a placeholder
    logging.info("Synthetic data generation completed.")


if __name__ == '__main__':
    
    load_dotenv()  # load environment variables from the .env file
    args = parse_arguments()  # parse command line arguments
    try:
        asyncio.run(main(args))            # run main function
    except KeyboardInterrupt:
        logging.info("Ctrl+C pressed!")
        sys.exit(0);
