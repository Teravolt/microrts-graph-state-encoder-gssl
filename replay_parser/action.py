"""
Convert unit and player actions found in the replays into symbolic actions
"""

UNIT_ACTION_ID_TO_NAME = {'0': 'idle', '1': 'move', '2': 'harvest',
                          '3': 'return', '4': 'produce', '5': 'attack'}
MOVE_ID_TO_DIRECTION_NAME = {'0': 'up', '1': 'right', '2': 'down', '3': 'left'}
UNIT_ACTION_LIST = ['idle', 'move', 'harvest', 'return', 'produce', 'attack']
UNIT_TYPE_LIST = ['worker', 'ranged', 'heavy', 'light', 'barracks', 'base']
DIRECTION_TYPE_LIST = ['up', 'down', 'left', 'right']

def create_unit_actions(trace_entry, unit_data_map: dict,
                        unit_actions_to_ignore: list,
                        allow_coordinates: bool = False):
    """
    Create a set of unit actions

    :param trace_entry: Set of unit actions
    :param unit_data_map: Information about each unit
    :param unit_actions_to_ignore: Unit action ids to ignore
    :param allow_coordinates: Allow coordinates in parameters

    :returns: Player id to set of executed unit actions
    """

    pid_to_unit_action_list = {}
    for primitive_act in trace_entry.find('actions'):

        unit_action = primitive_act.find('UnitAction')
        action_type = unit_action.attrib['type']

        if action_type in unit_actions_to_ignore:
            continue

        unit_id = primitive_act.attrib['unitID']
        player_id = int(unit_data_map[unit_id]['player'])

        if player_id not in pid_to_unit_action_list:
            pid_to_unit_action_list[player_id] = []

        act_name = UNIT_ACTION_ID_TO_NAME[action_type]
        unit_name = unit_data_map[unit_id]['type'].lower()
        unit_name = f"{unit_name}{unit_id}"

        parameters = {
            'unit-name': unit_name
            }

        if 'x' in unit_action.attrib.keys() and 'y' in unit_action.attrib.keys() \
            and allow_coordinates:
            parameters['x-coord'] = unit_action.attrib['x']
            parameters['y-coord'] = unit_action.attrib['y']

        if 'parameter' in unit_action.attrib.keys() \
            and unit_action.attrib['parameter'] != '10':
            direction = MOVE_ID_TO_DIRECTION_NAME[unit_action.attrib['parameter']]
            parameters['direction'] = direction

        if action_type == '4':
            unit_type_to_produce = unit_action.attrib['unitType'].lower()
            parameters['unit-produced'] = unit_type_to_produce

        pid_to_unit_action_list[player_id].append((act_name, parameters))

    return pid_to_unit_action_list
