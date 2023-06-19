"""
State representation for MicroRTS
"""

import numpy as np

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


def build_image_state(height, width, arena_map_as_str, unit_data_map):
    """
    Construct image representation of a state

    Args:
        height (int): Height of the game map
        width (int): Width of the game map
        arena_map_as_str (str): Game map in string representation
        unit_data_map (dict[str, obj]): Unit information

    Returns: ?
    """

    map_rows = list()
    total_num_tiles = width*height
    i = 0
    while i < total_num_tiles:
        row_str = arena_map_as_str[i:i+width]
        row = list()
        for j in range(0, len(row_str)):
            if row_str[j] != "1":
                row.append([0, 0, 0]) # r, g, b
            else:
                row.append([1, 1, 1])
        map_rows.append(row)
        i += width
    game_map = np.array(map_rows)  # (height, width, 3)
    player_map = np.array(map_rows)
    # print(game_map.shape, player_map.shape)
    for _, unit_data in unit_data_map.items():
        x = int(unit_data['x'])
        y = int(unit_data['y'])
        unit_type = unit_data["type"].lower()
        game_map[y, x, 0] = self.unit_type_to_color[unit_type][0]
        game_map[y, x, 1] = self.unit_type_to_color[unit_type][1]
        game_map[y, x, 2] = self.unit_type_to_color[unit_type][2]
        if unit_data["player"] != "-1":
            player_type = "player" if unit_data["player"] == player_id else "enemy"
            player_map[y, x, 0] = self.player_to_color[player_type][0]
            player_map[y, x, 1] = self.player_to_color[player_type][1]
            player_map[y, x, 2] = self.player_to_color[player_type][2]
    return (game_map, player_1_map, player_2_map)
