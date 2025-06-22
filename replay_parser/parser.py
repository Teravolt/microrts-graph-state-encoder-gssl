"""
Parse replay data and construct sequence of state-action pairs
"""

import argparse
from xml.etree import ElementTree
from zipfile import ZipFile

from copy import deepcopy

import os

import numpy as np

from tqdm import tqdm

from replay_parser.action import create_unit_actions
from replay_parser.action import create_one_hot_unit_actions
from replay_parser.state import build_simple_feature_vector_state, \
    build_image_state, build_graph_state, build_character_state

from replay_parser.utils.log_utils import LoggingUtils

def __add_state_action(state,
                       pid_to_action: dict,
                       player_to_trace: dict,
                       unit_actions_to_ignore: list,
                       is_graph=False,
                       is_2d_text=False):
    """
    Help add state-action pair to state-action traces

    :param pid_to_state: Player id to state
    :param pid_to_action: Player id to action
    :param player_to_trace: Player id to state-action trace
    """

    if is_graph:
        # Graph data structure will have list of players in it
        for i in range(len(state.players)):
            if i not in player_to_trace:
                player_to_trace[i] = []

            player_state = deepcopy(state)
            player_unit_mask = np.zeros(player_state.x.shape[0])
            for j, label in enumerate(player_state.labels):
                if i == label:
                    player_state.x[j, -1] = 1
                    player_unit_mask[j] = 1

            for j, action in enumerate(player_state.unit_actions):
                if str(int(action)) in unit_actions_to_ignore:
                    player_unit_mask[j] = 0

            player_state.player_unit_mask = player_unit_mask

            player_action = None if i not in pid_to_action else pid_to_action[i]
            player_to_trace[i].append((player_state, player_action))
    elif is_2d_text:
        for i in range(state.shape[0]):
            if i not in player_to_trace:
                player_to_trace[i] = []

            action = None if i not in pid_to_action else pid_to_action[i]

            player_state = state[i]
            player_to_trace[i].append((player_state, action))
    else:
        for i in range(state.shape[0]-1):
            if i not in player_to_trace:
                player_to_trace[i] = []

            action = None if i not in pid_to_action else pid_to_action[i]

            player_state = np.stack([state[i], state[-1]], axis=0)
            player_to_trace[i].append((player_state, action))


def parse_replay_xml(filename: str, config: dict):
    """
    Parse replay file in XML format
    
    :param filename: Name of XML file
    :param config: Config data for parser

    :returns: Player to state-action traces
    """

    read_from_zip = config.read_from_zip
    allow_coordinates = config.allow_coordinate
    unit_actions_to_ignore = config.unit_actions_to_ignore
    max_replay_length = config.max_replay_length
    frame_skip_freq = config.frame_skip_freq

    player_to_trace = {}
    player_ids = []

    root = None
    if read_from_zip:
        with ZipFile(filename, 'r') as zipf:
            xml_data = zipf.read('game.xml')
            root = ElementTree.fromstring(xml_data)
    else:
        tree = ElementTree.parse(filename)
        root = tree.getroot()

    assert root is not None

    # player_action_data = {}
    # player_action_data['variant_list'] = {}
    # player_action_data['type_list'] = {}
    # player_action_data['duplicate_index'] = 0

    for i, trace_entry in enumerate(root.find('entries')):
        # Assuming each entry here is a frame
        if i % frame_skip_freq != 0:
            continue

        if max_replay_length != -1 and i >= max_replay_length:
            break

        physical_game_state_entry = trace_entry.find('rts.PhysicalGameState')
        width = int(physical_game_state_entry.attrib['width'])
        height = int(physical_game_state_entry.attrib['height'])

        arena_map_str = physical_game_state_entry.find("terrain").text

        players_entry = physical_game_state_entry.find('players')
        for entry in players_entry:
            if entry.attrib['ID'] not in player_ids and entry.attrib['ID'] != '-1':
                if int(entry.attrib['ID']) not in player_ids:
                    player_ids.append(int(entry.attrib['ID']))

        unit_data_map = {}
        for unit in physical_game_state_entry.find('units'):
            unit_data_map[unit.attrib['ID']] = unit.attrib

        pid_to_action = create_unit_actions(
            trace_entry, unit_data_map,
            unit_actions_to_ignore,
            allow_coordinates=allow_coordinates)

        is_graph = False
        is_2d_text = False
        if config.state_representation == "graph":
            state = build_graph_state(unit_data_map)
            state.unit_actions = create_one_hot_unit_actions(pid_to_action, unit_data_map)
            is_graph = True
        elif config.state_representation == "feature":
            state = build_simple_feature_vector_state(unit_data_map)
        elif config.state_representation == "image":
            state = build_image_state(height, width, arena_map_str, unit_data_map)
        elif config.state_representation == "2d-text":
            state = build_character_state(height, width, arena_map_str, unit_data_map)
            is_2d_text = True
        else:
            raise ValueError(
                f'Unknown state representation: {config.state_representation}')

        __add_state_action(state,
                           pid_to_action,
                           player_to_trace,
                           unit_actions_to_ignore,
                           is_graph=is_graph,
                           is_2d_text=is_2d_text)

    player_to_trace = [
        (pid, player_to_trace[pid]) for pid in player_ids]

    return player_to_trace

def __parse_replay_dataset(replay_dataset: list, config: argparse.Namespace):
    """
    Help parse a set of replays

    
    :param input_directory: Path to replay dataset
    :param replay_dataset: List of replay filenames found at `input_directory`
    :param config: Config data

    :returns: Replay data
    """

    replay_data = []
    read_from_zip = config.read_from_zip

    for filename in tqdm(replay_dataset):

        fname_extension_removed = filename[0:filename.find(".xml")]

        if read_from_zip:
            fname_extension_removed = filename[0:filename.find(".zip")]

        file_path = f"{config.input_directory}/{filename}"
        player_to_trace = parse_replay_xml(file_path, config)

        for pid, trace in player_to_trace:
            replay_data.append((fname_extension_removed, pid, trace))

    return replay_data

def parse_replay_dataset(config: argparse.Namespace):
    """
    Parse a set of replays

    :param config: Config Data
    :returns: Replay data
    """

    input_directory = config.input_directory

    if input_directory is None:
        raise ValueError("Input Directory must be specified!")

    LoggingUtils.microrts_parser_logger.info(
        f"Extracting replay data from {input_directory}")
    replay_dataset = os.listdir(input_directory)

    read_from_zip = config.read_from_zip

    if read_from_zip:
        replay_dataset = [f for f in replay_dataset if ".zip" in f]
    else:
        replay_dataset = [f for f in replay_dataset if ".xml" in f]

    replay_dataset = [f for f in replay_dataset if ".swp" not in f]

    max_replays = config.max_replays if config.max_replays > 0 \
        else len(replay_dataset)

    LoggingUtils.microrts_parser_logger.info(
        f"Parsing {max_replays} replays!")

    replay_dataset = replay_dataset[0:max_replays]

    replay_data = __parse_replay_dataset(replay_dataset, config)

    return replay_data
