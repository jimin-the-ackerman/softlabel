
import yaml
import argparse


def write_argparse_args_to_yaml(args: argparse.Namespace,
                                filepath: str, 
                                exclude_keys: list[str] = ['output_dir', 'config']) -> None:
    """..."""
    
    args_dict = {
        k: v for k, v in vars(args).items() if k not in exclude_keys
    }
    with open(filepath, 'w') as file:
        yaml.dump(args_dict, file, default_flow_style=False)
