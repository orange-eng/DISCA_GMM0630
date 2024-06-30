import os, time, argparse, yaml


def parse_args_yaml(given_parser):
    new_parser = argparse.ArgumentParser()
    with open(given_parser.config_yaml, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        new_parser.set_defaults(**config)
        args = new_parser.parse_args()
        return args


def jupyter_parse_args_yaml(given_parser):
    new_parser = argparse.ArgumentParser()
    with open(given_parser.config_yaml, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        new_parser.set_defaults(**config)

        args = new_parser.parse_args(args=[])
        return args