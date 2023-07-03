import sys
import numpy as np

class information_parser():
    """
    class information_parser

    provide non-player specific information

    the following information are parsed:
        - map (updated)
        - map WIDTH (fixed)
        - map HEIGHT (fixed)
        - player id (fixed)
        - nbr of entities (updated)
        - player entities (updated)
        - bomb entities (updated)
        - item entities (updated)

    the following maps are updated:
        - item map
        - box map
        - static bomb map
        - dynamic bomb map
    """
    def __init__(self) -> None:
        # constants
        self.PATH:str = '.'
        self.WALL:str = 'X'
        self.BOX:str = '0'
        self.ITEM_EXTRA_RANGE:str = '1'
        self.ITEM_EXTRA_BOMB:str = '2'
        
        # constant weights
        self.DEFAULT_WEIGHT:int = 0
        self.OBSTACLE_WEIGHT:int = -9
        self.BOX_WEIGHT:int = 1

        # input variables (fixed)
        self.WIDTH, self.HEIGHT, self.MY_ID = [int(i) for i in input().split()]

        # input variables (updated)
        self.map:list = []
        self.entity_nbr:int = 0

        self.entity_player:list = []
        self.entity_bomb:list = []
        self.entity_item:list = []

        self.my_player:dict = {}

        # computed maps (updated)
        self.box_map:list = []
        self.item_map:list = []
        self.static_bomb_map:list = []
        self.dynamic_bomb_map:list = []

    def update_info(self) -> None:
        '''
        update the information
        '''
        def __set_entity(entity_type:int, owner:int, x:int, y:int, param_name_1:str, param_1:int, param_name_2:str, param_2:int):
            return { 'type': entity_type,
                    'owner': owner,
                    'x': x,
                    'y': y,
                    param_name_1: param_1,
                    param_name_2: param_2}
        # update map
        self.map = []

        for _ in range(self.HEIGHT):
            row = input()
            self.map.append(row)

        # update entity nbr
        self.entity_nbr = int(input())

        # update entity lists
        self.entity_player = []
        self.entity_bomb = []
        self.entity_item = []

        for _ in range(self.entity_nbr):
            entity_type, owner, x, y, param_1, param_2 = [int(j) for j in input().split()]
            if entity_type == 0:
                self.entity_player.append(
                    __set_entity(entity_type, owner, x, y,
                                 'nbr_bomb_left', param_1,
                                 'reach', param_2))
            elif entity_type == 1:
                self.entity_bomb.append(
                    __set_entity(entity_type, owner, x, y,
                                 'timer', param_1,
                                 'reach', param_2))
            elif entity_type == 2:
                self.entity_item.append(
                    __set_entity(entity_type, None, x, y,
                                 'item_type', param_1,
                                 'param_2', None))

        # update my player
        for player in self.entity_player:
            if player['owner'] == self.MY_ID:
                self.my_player = player
                break
        
        # update all maps
        dim:tuple = (self.HEIGHT, self.WIDTH)
        
        # update box map
        value_list = [(self.WALL, self.OBSTACLE_WEIGHT),
                      (self.BOX, self.BOX_WEIGHT),
                      (self.ITEM_EXTRA_RANGE, self.BOX_WEIGHT),
                      (self.ITEM_EXTRA_BOMB, self.BOX_WEIGHT)]
        self.box_map = self.create_map(self.map, dim, self.DEFAULT_WEIGHT, value_list)

        # update static bomb map
        value_list = [(self.WALL, self.OBSTACLE_WEIGHT),
                      (self.BOX, self.BOX_WEIGHT),
                      (self.ITEM_EXTRA_RANGE, self.BOX_WEIGHT),
                      (self.ITEM_EXTRA_BOMB, self.BOX_WEIGHT)]
        self.static_bomb_map = self.create_map(self.map, dim, self.DEFAULT_WEIGHT, value_list)
        
        for bomb in self.entity_bomb:
            reach:tuple = (bomb['reach'], -bomb['timer'])
            obstacle:list = [self.OBSTACLE_WEIGHT]
            affect_val:list = [(self.BOX_WEIGHT, -(bomb['timer']+1))]

            self.place_entity(self.static_bomb_map, dim, bomb, reach, obstacle, affect_val)


    def place_entity(self, from_map:list, dim:tuple, entity:dict, reach:tuple, obstacle:list, affect_val:list) -> None:
        '''
        Description
        ------
            place an entity on the given map\n
            the entity can be a bomb, a player or an item\n
            you can specify the reach of the bomb\n
            you can compute the the box affected by the bomb
        
        ----------------
        Param
        ------
            - `from_map`: the map to be updated `list[list[int]]`
            - `dim`: the dimension of the map `tuple(height:int, width:int)`
            - `entity`: the entity to be placed `dict: {type:int, owner:int, x:int, y:int, ...}`
            - `reach`: the reach of the bomb `tuple(reach_range:int, reach_val:int)`
            - `obstacle`: the obstacles to be affected by the bomb reach `list[int, ...]`
            - `affect_val`: the values to be affected by the bomb reach `list[(from_map value:any, affect_val:int), ...]`

        ----------------
        Return
        -------
            no return

        ----------------
        Raises
        ------
            no raise
        '''
        def __is_affectable(cur_cell:int, affect_val) -> bool:
            for val in affect_val:
                if cur_cell == val[0]:
                    return True
            return False
    
        def __get_affected_val(cur_cell:int, affect_val) -> int:
            for val in affect_val:
                if cur_cell == val[0]:
                    return val[1]
            return -69
    
        def __check_direction(direction:bool, from_map:list, coord:tuple, reach:tuple, obstacle:list, affect_val:list) -> bool:
            if from_map[coord[0]][coord[1]] in obstacle:
                return True
            elif not direction and __is_affectable(from_map[coord[0]][coord[1]], affect_val):
                from_map[coord[0]][coord[1]] = __get_affected_val(from_map[coord[0]][coord[1]], affect_val)
                return True
            elif not direction and from_map[coord[0]][coord[1]] not in range(reach[1], 0, 1):
                from_map[coord[0]][coord[1]] = reach[1]
            return False
        
        # place at center
        from_map[entity['y']][entity['x']] = reach[1]

        # place around
        if reach[0] > 0:
            up = False
            down = False
            left = False
            right = False

            for i in range(1, reach[0]):
                if entity['y']-i >= 0 and not up:
                    up = __check_direction(up, from_map, (entity['y']-i, entity['x']), reach, obstacle, affect_val)
                if entity['y']+i < dim[0] and not down:
                    down = __check_direction(down, from_map, (entity['y']+i, entity['x']), reach, obstacle, affect_val)
                if entity['x']-i >= 0 and not left:
                    left = __check_direction(left, from_map, (entity['y'], entity['x']-i), reach, obstacle, affect_val)
                if entity['x']+i < dim[1] and not right:
                    right = __check_direction(right, from_map, (entity['y'], entity['x']+i), reach, obstacle, affect_val)


    def create_map(self, from_map:list, dim:tuple, default_val:int, change_val:list) -> list:
        '''
        Description
        ------
            create a new map with weights from the given map

        ----------------
        Param
        ------
            - `from_map`: the map to be converted
            - `dim`: the dimension of the map `tuple(height:int, width:int)`
            - `default_val`: the default value of the map `int`
            - `change_val`: the values to be changed `list[(from_map value:any, weight:int), ...]`
        ----------------
        Return
        -------
            - `new_map`: the converted map
        ----------------
        Raises
        ------
            no raise
        '''
        new_map = np.full(dim, default_val)

        for i in range(dim[0]):
            for j in range(dim[1]):
                # change the value of the map with its corresponding weight
                for pair_val in change_val:
                    new_map[i][j] = np.where(from_map[i][j] == pair_val[0], pair_val[1], new_map[i][j])

        return new_map

    def print_elem(self, elem_name:str, elem) -> None:
        if type(elem) != list and type(elem) != np.ndarray:
            print(elem_name, elem, file=sys.stderr, flush=True)
        else:
            print(elem_name, file=sys.stderr, flush=True)
            for i in range(len(elem)):
                print(elem[i], file=sys.stderr, flush=True)

info = information_parser()

while True:
    info.update_info()

    #info.print_elem('map', info.map)
    #info.print_elem('entity_nbr', info.entity_nbr)
    #info.print_elem('entity_player', info.entity_player)
    info.print_elem('entity_bomb', info.entity_bomb)
    #info.print_elem('entity_item', info.entity_item)
    #info.print_elem('my_player', info.my_player)
    #info.print_elem('box_map', info.box_map)
    info.print_elem('static_bomb_map', info.static_bomb_map)

    print("MOVE 0 0")