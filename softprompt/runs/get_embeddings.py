import os
import sys
sys.path.insert(
    0, os.path.abspath("../../")
)

import argparse
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

from pathlib import Path
from dotenv import load_dotenv

import numpy as np
import pandas as pd

from langchain_openai import OpenAIEmbeddings


DEFAULT_TEXT_KEY = "text"  # Consider moving this to a constant configuration section
DEFAULT_JSONL_BASENAME = "data.jsonl"  # Renamed for clarity
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_EMBEDDING_BASENAME = "data.npy"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Compute vector embeddings of text using OpenAI API", add_help=True)
    parser.add_argument("-d", "--directory", type=str, required=True,
                        help=f"Directory containing the {DEFAULT_JSONL_BASENAME} file (REQUIRED)")
    parser.add_argument("--filename", type=str, default=None,
                        help=f"(Optional) Specify a custom filename instead of the default {DEFAULT_JSONL_BASENAME}")
    parser.add_argument("--batch_size", type=int, default=50,
                        help="Number of texts processed per asynchronous API call (default: 50)")
    parser.add_argument("--override", action="store_true",
                        help="Overwrite existing embeddings if the output file already exists (default: False)")
    return parser.parse_args()


def read_jsonl_data(filepath: str) -> pd.DataFrame:
        return pd.read_json(filepath, lines=True)


async def compute_openai_embeddings(text_list: list[str],
                                    batch_size: int = 50,
                                    model: str = DEFAULT_EMBEDDING_MODEL,
                                    ) -> np.ndarray:
    """
    Compute vector embeddings for a list of texts using the OpenAI API.

    Args:
        text_list (list[str]): List of text strings to embed.
        batch_size (int): Number of texts processed per batch.
        model (str): OpenAI embedding model to use.

    Returns:
        np.ndarray: Array of computed embeddings.
    """
    # Initialize the embeddings class
    embedder = OpenAIEmbeddings(model=model, max_retries=5, request_timeout=120.)

    # Create batches of texts
    batches = [text_list[i:i + batch_size] for i in range(0, len(text_list), batch_size)]
    logging.info(f"Split {len(text_list):,} documents into {len(batches):,} batches")

    embeddings = []
    
    batch_index_width: int = str(len(batches)).__len__()

    # Process each batch
    for i, batch in enumerate(batches):
        
        # Get embeddings for the current batch
        batch_embeddings: list = await embedder.aembed_documents(batch)
        embeddings.extend(batch_embeddings)
        logging.info(f"Completed batch {i+1:>{batch_index_width},}/{len(batches):>{batch_index_width},}")
        
        # Optional: Add a delay between batches to avoid rate limits
        if i < len(batches) - 1:
            await asyncio.sleep(0.5)  # 500ms delay between batches
    
    logging.info(f"Generated {len(embeddings):,} embeddings in total")

    return np.array(embeddings)


async def main(args: argparse.Namespace) -> None:
    """
    Main function to orchestrate the embedding computation process.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    jsonl_basename = args.filename or DEFAULT_JSONL_BASENAME
    embedding_model = DEFAULT_EMBEDDING_MODEL
    embedding_basename = DEFAULT_EMBEDDING_BASENAME

    # Check directory and filename
    directory = Path(args.directory).resolve()
    assert jsonl_basename in os.listdir(directory), f"{jsonl_basename} must exist in {directory}"

    # Before doing anything, check whether an embedding file already exists
    embedding_dir = directory / f"embeddings/openai/{embedding_model}"
    embedding_file = embedding_dir / embedding_basename
    if embedding_file.exists() and not args.override:
        raise FileExistsError(
            f"Embedding already exists: {embedding_file}\n Use the --override flag to ignore this."
        )

    # Read data
    df = read_jsonl_data(directory / jsonl_basename)
    logging.info(f"Successfully loaded data")
    
    # Compute embeddings
    embeddings = await compute_openai_embeddings(text_list=df[DEFAULT_TEXT_KEY], batch_size=args.batch_size)

    # Save the embeddings
    os.makedirs(embedding_dir, exist_ok=True)
    np.save(embedding_file, embeddings)
    logging.info(f"Saved embeddings to: {embedding_file}")
    

if __name__ == "__main__":

    load_dotenv()  # load environment variables from the .env file
    args = parse_arguments()  # parse command line arguments
    try:
        asyncio.run(main(args))  # run main function
    except KeyboardInterrupt:
        logging.info("Ctrl+C pressed!")
        sys.exit(0);
