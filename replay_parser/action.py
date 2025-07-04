"""
Unit or player action representations
"""

import numpy as np

UNIT_ACTION_ID_TO_NAME = {'0': 'idle', '1': 'move', '2': 'harvest',
                          '3': 'return', '4': 'produce', '5': 'attack'}
MOVE_ID_TO_DIRECTION_NAME = {'0': 'up', '1': 'right', '2': 'down', '3': 'left'}
UNIT_ACTION_LIST = ['idle', 'move', 'harvest', 'return', 'produce', 'attack']
UNIT_TYPE_LIST = ['worker', 'ranged', 'heavy', 'light', 'barracks', 'base']
DIRECTION_TYPE_LIST = ['up', 'down', 'left', 'right']

def create_unit_actions(trace_entry, unit_data_map: dict,
                        unit_actions_to_ignore: list):
    """
    Create a set of unit actions

    :param trace_entry: Set of unit actions
    :param unit_data_map: Information about each unit
    :param unit_actions_to_ignore: Unit action ids to ignore

    :returns: Player id to set of executed unit actions
    """

    pid_to_unit_actions = {}
    for unit_action_entry in trace_entry.find('actions'):

        unit_action = unit_action_entry.find('UnitAction')
        action_type = unit_action.attrib['type']

        if action_type in unit_actions_to_ignore:
            continue

        unit_id = unit_action_entry.attrib['unitID']
        player_id = int(unit_data_map[unit_id]['player'])

        if player_id not in pid_to_unit_actions:
            pid_to_unit_actions[player_id] = {}

        act_name = UNIT_ACTION_ID_TO_NAME[action_type]
        unit_type = unit_data_map[unit_id]['type'].lower()
        unit_name = f"{unit_type}{unit_id}"

        parameters = {
            'name': unit_name,
            'type': unit_type,
            'id': unit_id
            }

        if 'x' in unit_action.attrib and 'y' in unit_action.attrib:
            parameters['x'] = int(unit_action.attrib['x'])
            parameters['y'] = int(unit_action.attrib['y'])

        if 'parameter' in unit_action.attrib and act_name != "idle":
            direction = MOVE_ID_TO_DIRECTION_NAME[unit_action.attrib['parameter']]
            parameters['direction'] = direction

        if 'unitType' in unit_action.attrib:
            unit_type_to_produce = unit_action.attrib['unitType'].lower()
            parameters['unit-produced'] = unit_type_to_produce

        pid_to_unit_actions[player_id][unit_id] = (act_name, parameters)

    return pid_to_unit_actions

def create_unit_actions_matrix(unit_states: dict,
                               unit_actions: dict,
                               unit_type_table: dict):
    """
    Create one hot encoding of actions per unit

    :param unit_states: State information about each unit
    :param unit_actions: Unit actions
    :param unit_type_table: Table containg info about each unit type
    :returns: Matrix of size UxN, where U is the number of units in the state
    and N is the unit action feature vector
    """

    max_attack_range = max([int(specs['attackRange']) for specs in unit_type_table.values()])
    max_attack_range = 2*max_attack_range + 1

    action_features = len(UNIT_ACTION_LIST) + 4*len(DIRECTION_TYPE_LIST) + \
        len(UNIT_TYPE_LIST) + max_attack_range*max_attack_range

    unit_action_matrix = np.zeros((len(unit_states), action_features))

    # action type, move direction, harvest direction, return direction, produce direction,
    # produce type, relative attack position

    # action_type = 6 len(UNIT_ACTION_LIST)
    # move direction = 4 len(DIRECTION_TYPE_LIST)
    # harvest direction = 4 len(DIRECTION_TYPE_LIST)
    # return direction = 4 len(DIRECTION_TYPE_LIST)
    # produce direction = 4 len(DIRECTION_TYPE_LIST)
    # produce type = 6 len(UNIT_TYPE_LIST)
    # relative attack position = max-attack-range^2

    # Add one-hot encoding of actions per unit
    for j, (unit_id, state) in enumerate(unit_states.items()):
        if unit_id in unit_actions:
            (name, parameters) = unit_actions[unit_id]
            action_idx =  UNIT_ACTION_LIST.index(name)
            assert action_idx >= 0
            unit_action_matrix[j][action_idx] = 1

            if 'x' in parameters and 'y' in parameters:
                offset = len(UNIT_ACTION_LIST) + 4*len(DIRECTION_TYPE_LIST) + len(UNIT_TYPE_LIST)
                x_attack = parameters['x']
                y_attack = parameters['y']
                x = int(state['x'])
                y = int(state['y'])
                center_coord = max_attack_range//2
                x_rel = x_attack-x
                y_rel = y_attack-y
                # print(f"Attack pos: ({x_attack},{y_attack}) - Current pos: ({x},{y})")
                # print(f"Max attack range: {max_attack_range}, Grid pos: ({center_coord+x_rel},{center_coord+y_rel})")
                # print(f"Relative position: ({x_rel},{y_rel})")
                # print(f"Center coord: {center_coord}")
                attack_idx = (center_coord+y_rel)*max_attack_range+(center_coord+x_rel)
                unit_action_matrix[j][offset + attack_idx] = 1

            if 'direction' in parameters and name != 'idle':
                offset = -1
                direction_idx = DIRECTION_TYPE_LIST.index(parameters['direction'])

                if name == 'move':
                    offset = len(UNIT_ACTION_LIST)
                elif name == 'harvest':
                    offset = len(UNIT_ACTION_LIST) + len(DIRECTION_TYPE_LIST)
                elif name == 'return':
                    offset = len(UNIT_ACTION_LIST) + 2*len(DIRECTION_TYPE_LIST)
                elif name == 'produce':
                    offset = len(UNIT_ACTION_LIST) + 3*len(DIRECTION_TYPE_LIST)

                assert offset != -1
                unit_action_matrix[j][offset + direction_idx] = 1

            if 'unit-produced' in parameters:
                unit_offset = len(UNIT_ACTION_LIST) + 4*len(DIRECTION_TYPE_LIST)
                unit_idx = UNIT_TYPE_LIST.index(parameters['unit-produced'])
                unit_action_matrix[j][unit_offset + unit_idx] = 1

            # print(f"Action name: {name}, Parameters: {parameters}")
            # print(f"Action vector: {unit_action_matrix[j]}")

    return unit_action_matrix
