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

import numpy as np

from typing import List, Tuple, Dict, Optional, Iterable

from rich.logging import RichHandler
from rich.console import Console
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_core.exceptions import OutputParserException
from pydantic_core._pydantic_core import ValidationError

from softprompt.templates import data_to_template
from softprompt.schemas import get_output_schema

from softprompt.utils import generate_unique_identifier
from softprompt.utils import write_argparse_args_to_yaml
from softprompt.utils import temporary_log_level
from softprompt.utils import write_chat_template_to_jsonl
from softprompt.utils import get_model_provider


DATA_TO_LABELS = {
    "emotion": ["sadness", "joy", "love", "anger", "fear", "surprise"],
    "agnews": ["World", "Sports", "Business", "Sci/Tech"],  # TODO: 
}

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[RichHandler()],
)


# https://www.reddit.com/r/LocalLLaMA/comments/1h5plai/does_think_step_by_step_work_with_structured/
CHAIN_OF_THOUGHT = "\n".join(
    [
        "Before writing your final answer, please think step-by-step.",
        "Ensure that you MUST provide your step-by-step thinking process in the final output.",
    ]
)


def parse_arguments():
    parser = argparse.ArgumentParser('Generating synthetic emotion data using LLMs...')
    parser.add_argument('--data', type=str, default='emotion',
                        choices=['emotion', 'agnews', 'nyt'],
                        help='Dataset name')
    parser.add_argument('--model', type=str, default='gemini-2.0-flash',
                        choices=['gpt-4o-mini',
                                 'o1',
                                 'o3-mini',
                                 'gemini-2.0-flash',
                                 'gemini-2.5-flash-preview-04-17',
                                 ],
                        help='LLM model')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='LLM temperature')
    parser.add_argument('--max_tokens', type=int, default=None,
                        help='Maximum number of LLM-generated tokens')
    
    parser.add_argument('--api_key', type=str, default=None)
    
    parser.add_argument('--strategy', type=str, default='very spiky',
                        choices=('very spiky', 'spikier', 'slightly spiky', 'flat', 'very flat', 'mixed'),
                        help="Label distribution strategy")

    parser.add_argument('--hard', action='store_true', help="Generate hard labels")
    parser.add_argument('--cot', action='store_true', help="Use chain of thought")

    parser.add_argument('--batch_size', type=int, default=50, help="Batch size for API calls")

    parser.add_argument('--sample_size', type=int, default=10000, help='Number of samples to generate')
    parser.add_argument('--log_interval', type=int, default=2, help='Logging interval')
    parser.add_argument('--output_dir', type=str, default='results/', help='Output directory')
    parser.add_argument('--config', action='append')
    args = parser.parse_args()
    if args.config:
        for filename in args.config:
            with open(filename, 'r') as f:
                cfg = yaml.safe_load(f)
                parser.set_defaults(**cfg)
    return parser.parse_args()


def sample_hard_label(num_classes: int) -> np.ndarray:
    return np.eye(num_classes)[random.randrange(0, num_classes)]


def sample_soft_label(num_classes: int,
                      strategy: str = "mixed",
                      ) -> np.ndarray:
    alpha_options = [
        [0.1] * num_classes,  # very spiky
        [0.3] * num_classes,  # spikier
        [0.5] * num_classes,  # slightly spiky
        [1.0] * num_classes,  # balanced        
        [5.0] * num_classes,  # very flat/high entropy
    ]
    if strategy == "very spiky":
        alpha = alpha_options[0]
    elif strategy == "spikier":
        alpha = alpha_options[1]
    elif strategy == "slightly spiky":
        alpha = alpha_options[2]
    elif strategy == "flat":
        alpha = alpha_options[3]
    elif strategy == "very flat":
        alpha = alpha_options[4]
    elif strategy == "mixed":
        alpha = random.choice(alpha_options)
    else:
        raise ValueError
    return np.random.dirichlet(alpha)


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
    logging.info(f"Chat model: {llm.__class__.__name__}")

    # get chat template
    template = data_to_template[args.data]  # a `ChatPromptTemplate` instance

    # fill in static variables
    template = template.partial(
        chain_of_thought_instructions=CHAIN_OF_THOUGHT if args.cot else "",
    )

    template_output_file = os.path.join(output_dir, 'template.jsonl')
    write_chat_template_to_jsonl(template, filepath=template_output_file)
    logging.info(f"Template saved to: {template_output_file}")

    # structured output
    schema = get_output_schema(data=args.data, cot=args.cot)
    structured_llm = llm.with_structured_output(schema)
    logging.info(f"Output schema: {schema.__name__}")

    # create a runnable chain
    chain = template | structured_llm

    with temporary_log_level(logging.INFO):
        
        class_names: list[str] = DATA_TO_LABELS[args.data]
        k = len(class_names)

        # generate synthetic data
        i: int = 0
        count: int = 0
        while count < args.sample_size:
            
            if args.hard:
                scores: list[np.ndarray] = [
                    sample_hard_label(k) for _ in range(args.batch_size)
                ]
            else:
                scores: list[np.ndarray] = [
                    sample_soft_label(k, strategy=args.strategy) for _ in range(args.batch_size)
                ]

            processed_scores: List[Dict[str,float]] = []  # list of dicts
            for score in scores:
                processed_scores.append(
                    {c: float(round(s, 3)) for c, s in zip(class_names, score)}
                )

            try:
                # calling api
                chain_inputs: List[Dict[str,Dict]] = [
                    {'score': s} for _, s in enumerate(processed_scores)
                ]
                responses: list = await chain.abatch(
                    chain_inputs, config={"max_concurrency": args.batch_size}
                )

                # write a formatted version of the template for backtracking
                formatted_msg_file = os.path.join(output_dir, "template_formatted.txt")
                if (i != 0) and not os.path.exists(formatted_msg_file):
                    formatted_messages = template.format_messages(**chain_inputs[0])
                    with open(formatted_msg_file, 'w') as file:
                        for msg in formatted_messages:
                            file.write(f"{msg.type}:\n")
                            file.write(f"{msg.content}\n")

            except (OutputParserException,
                    ValidationError,
                    json.JSONDecodeError) as e:
                logging.warning(f"Exception raised (i={i}). Retrying...")
                continue

            # write result to file
            outfile: str = os.path.join(output_dir, 'data.jsonl')
            with open(outfile, 'a') as f:
                invalid_count = 0
                for response in responses:
                    try:
                        json_line = json.dumps(response.model_dump())
                        f.write(json_line + "\n")
                    except AttributeError:
                        invalid_count += 1
                        continue

            count += len(responses)
            count -= invalid_count

            if (i+1) % args.log_interval == 0:
                with temporary_log_level(logging.INFO):
                    logging.info(f"Generated {count:>6,}/{args.sample_size:>6,} examples")
            
            time.sleep(0.5)
            i += 1;

    with open(os.path.join(output_dir, "main.log"), "w") as f:
        pass
    logging.info("End of program")


if __name__ == '__main__':
    load_dotenv()
    args = parse_arguments()
    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        logging.info("Ctrl+C pressed!")
        sys.exit(0);
