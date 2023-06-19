"""
Example usage of microrts replay parser
"""

import argparse

from replay_parser.parser import parse_replay_dataset

def get_config():
    """
    Build config

    :returns: Config from command line arguments
    """

    parser = argparse.ArgumentParser(description='MicroRTS Analyzer')

    parser.add_argument('input_directory', help='Directory containing replays')
    parser.add_argument('--read_from_zip', action='store_true',
                        help='Read from zip file')
    parser.add_argument('--allow_coordinate', action='store_true',
                        help='Allow coordinates in states and actions')
    parser.add_argument('--unit_actions_to_ignore', type=list, default=[],
                        help='Unit actions to ignore')
    parser.add_argument('--max_replays', default=-1, type=int,
                        help='Maximum number of replays to read')
    parser.add_argument('--max_replay_length', default=-1, type=int,
                        help='Maximum replay length')
    parser.add_argument('--seed', default=1, type=int, help='Random seed')

    config = parser.parse_args()

    return config

def main():
    """
    Main function
    """

    config = get_config()
    replay_data = parse_replay_dataset(config)

if __name__ == "__main__":
    main()
