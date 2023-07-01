"""
State representation for MicroRTS
"""

import numpy as np

import torch
from torch import nn

UNIT_TYPE_TO_COLOR = {
    "worker": (128, 128, 128, 1.0),
    "base": (255, 255, 255, 1.0),
    "barracks": (192, 192, 192, 1.0),
    "heavy": (255, 255, 0, 1.0),
    "light": (255, 0, 255, 1.0),
    "ranged": (0, 255, 255, 1.0),
    "resource": (0, 255, 0, 1.0)
    }

PLAYER_TO_COLOR = {
    "player": (0, 0, 255, 1.0),
    "enemy": (255, 0, 0, 1.0)
    }

NUM_PLAYER_FEATURES = 14
NUM_GAME_FEATURES = 2
NUM_STATE_FEATURES = NUM_PLAYER_FEATURES + NUM_GAME_FEATURES

SCALE_FACTOR = 8
UPSAMPLER = nn.Upsample(
    scale_factor=SCALE_FACTOR,
    mode='nearest')

def build_simple_feature_vector_state(unit_data_map: dict):
    """
    Construct a simple state representation (vector of features)

    Notes:
        feature vector is a concatenation of player + enemy vectors
        player/enemy vectors: [n worker, n light, n heavy, n ranged, n base, n barracks,
            worker health, light health, heavy health, ranged health,
            base health, barracks health,
            worker resources, base resources]
        Game features: [height*width, number of non-player resources]

    :param height: Height of game map
    :param width: Width of game map
    :param unit_data_map: Information about each unit in the game state

    :returns: State features
    """

    ordered_unit_types = [
        'worker',
        'light',
        'heavy',
        'ranged',
        'barracks',
        'base']

    players = []

    for _, unit_data in unit_data_map.items():
        player_id = int(unit_data['player'])
        if player_id != -1:
            if player_id not in players:
                players.append(player_id)

    state_features = np.zeros((len(players)+1, NUM_PLAYER_FEATURES))

    for _, unit_data in unit_data_map.items():
        unit_type = unit_data['type'].lower()
        player = int(unit_data['player'])

        if player == -1:
            state_features[-1][1] += float(unit_data['resources'])
            continue

        index = ordered_unit_types.index(unit_type)
        state_features[player][index] += 1
        state_features[player][index + 6] += float(unit_data['hitpoints'])

        if unit_type == 'worker':
            state_features[player][12] += float(unit_data['resources'])
        elif unit_type == 'base':
            state_features[player][13] += float(unit_data['resources'])
        else:
            continue

    return state_features


def upsample_state(state):
    """
    Upsample state
    """

    upsampled_state = torch.tensor(state)
    upsampled_state = upsampled_state.transpose(0, 2).unsqueeze(0)
    upsampled_state = UPSAMPLER(upsampled_state).squeeze(0)
    upsampled_state = upsampled_state.transpose(0, 2)

    upsampled_state = upsampled_state.numpy()

    return upsampled_state


def build_image_state(height: int, width: int,
                      arena_map_str,
                      unit_data_map):
    """
    Construct image representation of a state

    :param height: Height of the game map
    :param width: Width of the game map
    :param arena_map_str: Game map in string representation
    :param unit_data_map: Unit information

    :returns: State features as images
    """

    players = []

    for _, unit_data in unit_data_map.items():
        player_id = int(unit_data['player'])
        if player_id != -1:
            if player_id not in players:
                players.append(player_id)

    game_map = np.zeros((height, width, 3))

    total_num_tiles = width*height
    i = 0

    while i < total_num_tiles:
        row_str = arena_map_str[i:i+width]

        for j, elem in enumerate(row_str):
            if elem != '1':
                game_map[i // width, j, :] = 0
                # row.append([0, 0, 0]) # r, g, b
            else:
                game_map[i // width, j, :] = 1
                # row.append([1, 1, 1])

        i += width

    state_features = np.stack([np.array(game_map)]*(len(players)+1))

    for _, unit_data in unit_data_map.items():
        x = int(unit_data['x'])
        y = int(unit_data['y'])

        unit_type = unit_data['type'].lower()

        state_features[-1, y, x, 0] = UNIT_TYPE_TO_COLOR[unit_type][0]
        state_features[-1, y, x, 1] = UNIT_TYPE_TO_COLOR[unit_type][1]
        state_features[-1, y, x, 2] = UNIT_TYPE_TO_COLOR[unit_type][2]

        if unit_data['player'] != '-1':
            player_id = int(unit_data['player'])

            state_features[player_id, y, x, 0] = PLAYER_TO_COLOR['player'][0]
            state_features[player_id, y, x, 1] = PLAYER_TO_COLOR['player'][1]
            state_features[player_id, y, x, 2] = PLAYER_TO_COLOR['player'][2]

            for pid in players:
                if pid != player_id:
                    state_features[pid, y, x, 0] = PLAYER_TO_COLOR['enemy'][0]
                    state_features[pid, y, x, 1] = PLAYER_TO_COLOR['enemy'][1]
                    state_features[pid, y, x, 2] = PLAYER_TO_COLOR['enemy'][2]

    upsampled_game_state = upsample_state(state_features[-1])

    player_states = []
    for pid in players:
        upsampled_state = upsample_state(state_features[pid])
        player_states.append(upsampled_state)

    state = np.stack(player_states + [upsampled_game_state], axis=0)

    return state
