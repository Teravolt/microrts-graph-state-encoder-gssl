"""
Convert unit and player actions found in the replays into symbolic actions
"""

UNIT_ACTION_ID_TO_NAME = {'0': 'idle', '1': 'move', '2': 'harvest',
                          '3': 'return', '4': 'produce', '5': 'attack'}
MOVE_ID_TO_DIRECTION_NAME = {'0': 'up', '1': 'right', '2': 'down', '3': 'left'}
UNIT_ACTION_LIST = ['idle', 'move', 'harvest', 'return', 'produce', 'attack']
UNIT_TYPE_LIST = ['worker', 'ranged', 'heavy', 'light', 'barracks', 'base']
DIRECTION_TYPE_LIST = ['up', 'down', 'left', 'right']

ACTION_FORMAT = lambda name, param_list: "{}({})".format(name, ",".join(param_list))

def get_types(object_list: list):
    """Get type list

    Args:
        object_list (list[str]): Object list

    Returns:
        type_list (list[str]): Type list
    """

    type_list = list()
    for obj in object_list:
        try:
            int(obj)
            type_list.append("coord")
        except ValueError:
            continue
        if obj in DIRECTION_TYPE_LIST:
            type_list.append("direction")
            continue
        if obj in UNIT_TYPE_LIST:
            type_list.append("unittype")
            continue
        for unit in UNIT_TYPE_LIST:
            if unit in obj and unit != obj:
                type_list.append("unit")
                break
    return type_list

def get_action_name_and_parameters(unit_action_list: list, params_ordered=False,
                                   params_unique=False):
    """Construct action name and two lists of parameters. One is a list of objects
    and the other is a list of types of each object

    Args:
        unit_action_list (list[str]): List of unit actions in player action
        params_ordered (bool): True if parameters are to be ordered
        params_unique (bool): True if object parameters in player action are unique

    Returns:
        player_action_name (str): Player Action name
        object_list (list[str]): List of objects
        type_list (list[str]): List of types of each object in `object_list`
    """

    unit_action_list.sort(key=(lambda act: act[0:act.find("(")]))
    object_list = list()
    action_name_list = list()
    for act in unit_action_list:
        action_name = act[0:act.find("(")]
        param_list = act[act.find("(")+1:act.rfind(")")].split(",")
        for obj in param_list:
            if params_unique and obj in object_list:
                continue
            object_list.append(obj)
        action_name_list.append(action_name)

    player_action_name = "_".join(action_name_list)
    type_list = get_types(object_list)
    # for obj in object_list:
    #     if obj in DIRECTION_TYPE_LIST:
    #         type_list.append("direction")
    #         continue
    #     if obj in UNIT_TYPE_LIST:
    #         type_list.append("unittype")
    #         continue
    #     for unit in UNIT_TYPE_LIST:
    #         if unit in obj and unit != obj:
    #             type_list.append("unit")
    #             break

    if params_ordered:
        type_object_pairs = list(zip(type_list, object_list))
        type_object_pairs = sorted(type_object_pairs)
        object_list = [x for _, x in type_object_pairs]
        type_list = [x for x, _ in type_object_pairs]

    return player_action_name, object_list, type_list

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

        # action = act_name + f"({unit_name})"
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

        # if "x" in unit_action.attrib.keys() and "y" in unit_action.attrib.keys() \
        #     and allow_coordinates:
        #     action = ",".join([unit_name, unit_action.attrib['x'], unit_action.attrib['y']])
        #     action = f"{act_name}({action})"

        # if 'parameter' in unit_action.attrib.keys() \
        #     and unit_action.attrib['parameter'] != "10":
        #     direction = MOVE_ID_TO_DIRECTION_NAME[unit_action.attrib['parameter']]
        #     action = ",".join([unit_name, direction])
        #     action = f"{act_name}({action})"

        # if action_type == '4':
        #     unit_type_to_produce = unit_action.attrib['unitType'].lower()
        #     direction = MOVE_ID_TO_DIRECTION_NAME[unit_action.attrib['parameter']]
        #     action = ",".join([unit_name, direction, unit_type_to_produce])
        #     action = f"{act_name}({action})"

    return pid_to_unit_action_list

# def create_symbolic_player_action(pid_to_unit_action_list: dict,
#                                   player_action_data: dict, config: dict):
#     """Create symbolic player action

#     Args:
#         pid_to_unit_action_list (dict[str, list[list[str]]]): Player id to set of
#     executed unit actions
#         player_action_data (dict[str, obj]): Data pertaining to player actions
#         config (dict[str, obj]): Config data

#     Returns:
#         player_to_action (dict[str, list[str]]): Player id to action executed by
#     player
#     """

#     player_to_action = dict()

#     is_first_order = config.get("is_first_order", True)

#     for player_id, unit_action_list in pid_to_unit_action_list.items():
#         if len(unit_action_list) == 0:
#             continue

#         player_action_name, object_list, type_list = \
#             get_action_name_and_parameters(unit_action_list)

#         player_action_name = "{}_args_{}".format(player_action_name, len(type_list))

#         if player_action_name not in player_action_data["variant_list"]:
#             player_action_data["variant_list"][player_action_name] = list()

#         if player_action_name not in player_action_data["type_list"]:
#             player_action_data["type_list"][player_action_name] = type_list
#             player_action_data["variant_list"][player_action_name].append(player_action_name)
#         else:
#             # Found a duplicate player action name.
#             # Check to see if the player action type is already used
#             is_found = False
#             for _player_action_name in player_action_data["variant_list"][player_action_name]:
#                 variant_type_list = player_action_data["type_list"][_player_action_name]
#                 if sorted(variant_type_list) == sorted(type_list):
#                     player_action_name = _player_action_name
#                     is_found = True
#                     break
#             if not is_found:
#                 _player_action_name = "{}_{}".format(
#                     player_action_name,
#                     player_action_data["duplicate_index"])

#                 player_action_data["variant_list"][player_action_name].append(
#                     _player_action_name)

#                 player_action_name = _player_action_name
#                 player_action_data["type_list"][player_action_name] = type_list
#                 player_action_data["duplicate_index"] += 1

#         action = ACTION_FORMAT(
#             player_action_name,
#             object_list if is_first_order else [])

#         # action_type = ACTION_FORMAT(
#         #     player_action_name,
#         #     type_list if is_first_order else [])
#         # if action_type not in player_action_data["action_types"]:
#         #     player_action_data["action_types"].append(action_type)

#         player_to_action[player_id] = action
#     return player_to_action
