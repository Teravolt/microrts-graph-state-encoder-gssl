"""
State representation for MicroRTS
"""

import numpy as np

from scipy.spatial.distance import euclidean

import torch
from torch import nn

from torch_geometric.data import Data

from transformers import AutoTokenizer, AutoModelForMaskedLM

UNIT_TYPE_TO_COLOR = {
    "worker": (128, 128, 128, 1.0),
    "base": (255, 255, 255, 1.0),
    "barracks": (192, 192, 192, 1.0),
    "heavy": (255, 255, 0, 1.0),
    "light": (255, 0, 255, 1.0),
    "ranged": (0, 255, 255, 1.0),
    "resource": (0, 255, 0, 1.0)
    }

UNIT_TYPES = list(UNIT_TYPE_TO_COLOR.keys())

PLAYER_TO_COLOR = {
    "player": (0, 0, 255, 1.0),
    "enemy": (255, 0, 0, 1.0)
    }

UNIT_TYPE_TO_ID = {
    "worker": 2,
    "base": 3,
    "barracks": 4,
    "heavy": 5,
    "light": 6,
    "ranged": 7,
    "resource": 8
}
ID_TO_UNIT_TYPE = {
    0: "empty",
    1: "wall",
    2: "worker",
    3: "base",
    4: "barracks",
    5: "heavy",
    6: "light",
    7: "ranged",
    8: "resource"
}

UNIT_TYPE_ID_TO_CHARACTER = {
    'empty': '-',
    'wall': "|",
    "worker": 'W',
    "base": '@',
    "barracks": '#',
    "heavy": 'H',
    "light": 'L',
    "ranged": 'R',
    "resource": '$'
}

NUM_PLAYER_FEATURES = 14
NUM_GAME_FEATURES = 2
NUM_STATE_FEATURES = NUM_PLAYER_FEATURES + NUM_GAME_FEATURES

SCALE_FACTOR = 8
UPSAMPLER = nn.Upsample(
    scale_factor=SCALE_FACTOR,
    mode='nearest')

DEVICE = torch.device(
    'cuda' if torch.cuda.is_available() \
        else 'mps' if torch.backends.mps.is_available() else 'cpu')

MODEL_ID = 'facebook/bart-large'
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_ID)
TEXT_EMBEDDING_MODEL = AutoModelForMaskedLM.from_pretrained(MODEL_ID).to(DEVICE)

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

def build_graph_state(unit_data_map):
    """
    Construct a graph-representation of a state

    :param unit_data_map: Unit information
    """

    # Considering a fully-connected graph
    # Node are entities and resources on the map
    # Edges are distance relationship between each
    # entity/resource

    num_nodes = len(unit_data_map)
    num_edges = len(unit_data_map)*len(unit_data_map)

    players = []

    for _, unit_data in unit_data_map.items():
        player_id = int(unit_data['player'])
        if player_id != -1:
            if player_id not in players:
                players.append(player_id)

    labels = []
    num_features = len(UNIT_TYPES) + 3
    node_features = np.zeros((num_nodes, num_features))

    edge_index = np.zeros((2, num_edges))
    edge_attr = np.zeros((num_edges, 2))

    edge_idx = 0
    for i, (_, unit_data) in enumerate(unit_data_map.items()):
        unit_type = unit_data['type'].lower()

        player_id = int(unit_data['player'])
        # print(f"Unit data: {unit_data}")

        node_features[i, UNIT_TYPES.index(unit_type)] = 1
        node_features[i, len(UNIT_TYPES)] = float(unit_data['resources'])
        node_features[i, len(UNIT_TYPES)+1] = float(unit_data['hitpoints'])
        labels.append(player_id)

        # if player_id != -1:
        #     node_features[i, len(UNIT_TYPES)+ 2 + players.index(player_id)] = 1
        # else:
        #     node_features[i, -2] = 1

        unit_coord_1 = float(unit_data['x']), float(unit_data['y'])
        for j, (_, unit_data) in enumerate(unit_data_map.items()):
            if i == j:
                # No self-loops
                continue

            unit_coord_2 = float(unit_data['x']), float(unit_data['y'])
            edge_index[0][edge_idx] = i
            edge_index[1][edge_idx] = j
            edge_attr[edge_idx][0] = euclidean(unit_coord_1, unit_coord_2)

            y_diff = unit_coord_2[1]-unit_coord_1[1]
            x_diff = unit_coord_2[0]-unit_coord_1[0]
            angle_rad = 0
            if x_diff != 0:
                angle_rad = np.arctan(y_diff/x_diff)
            if y_diff > 0 and x_diff < 0:
                # Quadrant 2
                # print("Quadrant 2")
                angle_rad = np.pi-angle_rad
            elif y_diff < 0 and x_diff < 0:
                # Quadrant 3
                # print("Quadrant 3")
                angle_rad = np.pi+angle_rad
            elif y_diff < 0 and x_diff > 0:
                # Quadrant 4
                # print("Quadrant 4")
                angle_rad = 2*np.pi-angle_rad

            edge_attr[edge_idx][1] = angle_rad
            edge_idx += 1

    graph = Data(x=torch.tensor(node_features, dtype=torch.float32),
                 edge_index=torch.tensor(edge_index, dtype=torch.int64),
                 edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
                 players=players,
                 labels=labels)

    return graph

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
        else:
            for pid in players:
                state_features[pid, y, x, 0] = UNIT_TYPE_TO_COLOR[unit_type][0]
                state_features[pid, y, x, 1] = UNIT_TYPE_TO_COLOR[unit_type][1]
                state_features[pid, y, x, 2] = UNIT_TYPE_TO_COLOR[unit_type][2]

    # upsampled_game_state = upsample_state(state_features[-1])
    upsampled_game_state = state_features[-1]

    player_states = []
    for pid in players:
        # upsampled_state = upsample_state(state_features[pid])
        player_states.append(state_features[pid])

    state = np.stack(player_states + [upsampled_game_state], axis=0)

    return state

def build_character_state(height: int, width: int,
                          arena_map_str,
                          unit_data_map):
    """
    Build 2D grid with characters

    :param height: Height of the game map
    :param width: Width of the game map
    :param arena_map_str: Game map in string representation
    :param unit_data_map: Unit information
    """

    players = []

    for _, unit_data in unit_data_map.items():
        player_id = int(unit_data['player'])
        if player_id != -1:
            if player_id not in players:
                players.append(player_id)

    game_map = np.zeros((height, width))

    total_num_tiles = width*height
    i = 0

    while i < total_num_tiles:
        row_str = arena_map_str[i:i+width]

        for j, elem in enumerate(row_str):
            if elem != '1':
                game_map[i // width, j] = 0
                # row.append([0, 0, 0]) # r, g, b
            else:
                game_map[i // width, j] = 1
                # row.append([1, 1, 1])

        i += width

    max_player_id = max(players)
    base_state_features = np.stack([np.array(game_map)]*(max_player_id+1))
    # print(f"State feature shape: {state_features.shape}")
    # print(f"Players: {players}")

    for _, unit_data in unit_data_map.items():
        x = int(unit_data['x'])
        y = int(unit_data['y'])

        unit_type = unit_data['type'].lower()
        player_id = int(unit_data['player'])

        if player_id == -1:
            for i in range(base_state_features.shape[0]):
                base_state_features[i, y, x] = UNIT_TYPE_TO_ID[unit_type]
        else:
            base_state_features[player_id, y, x] = UNIT_TYPE_TO_ID[unit_type]

    state_features = []
    for player_id in range(base_state_features.shape[0]):
        batch_text = []
        for x in range(base_state_features[player_id].shape[1]):
            text = []
            for y in range(base_state_features[player_id].shape[0]):
                token = UNIT_TYPE_ID_TO_CHARACTER[ID_TO_UNIT_TYPE[int(base_state_features[player_id,y,x])]]
                text.append(token)

            batch_text.append(', '.join(text))

        with torch.no_grad():
            model_input = TOKENIZER(batch_text, return_tensors="pt",
                                    padding='max_length', truncation=True)
            # print(model_input['input_ids'].shape)
            output = TEXT_EMBEDDING_MODEL(**model_input)

            last_hidden_state = output.encoder_last_hidden_state
            last_hidden_state = last_hidden_state.mean(dim=1)
            # print(f"Mean of second dimension: {last_hidden_state.shape}")
            last_hidden_state = last_hidden_state.mean(dim=0)
            # print(f"Mean of first dimension: {last_hidden_state.shape}")

            # print(f"Player ID: {player_id}; State Embedding: {state_embedding.shape}")
            state_features.append(last_hidden_state)

    state_features = torch.stack(state_features, dim=0)
    return state_features

# def build_text_state(config: Namespace):
#     """
#     Construct text representation of a state

#     :param height: Height of the game map
#     :param width: Width of the game map
#     :param arena_map_str: Game map in string representation
#     :param unit_data_map: Unit information

#     :returns: Text embedding as state
#     """

#     # TODO: How can we represent the state as text (i.e., free-form text)

#     # Map Information
#     # Unit Information
    