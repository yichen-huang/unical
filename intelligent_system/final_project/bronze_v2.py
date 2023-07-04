from calendar import c
import sys
import numpy as np

class information_parser():
    """
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
        - box map
        - static bomb map
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
        #self.item_map:list = []
        self.static_bomb_map:list = []
        #self.dynamic_bomb_map:list = []

    def update_info(self) -> None:
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
        
        # update box map
        val_list:tuple = [(self.WALL, self.OBSTACLE_WEIGHT),
                      (self.BOX, self.BOX_WEIGHT),
                      (self.ITEM_EXTRA_RANGE, self.BOX_WEIGHT),
                      (self.ITEM_EXTRA_BOMB, self.BOX_WEIGHT)]
        self.box_map = self.create_map(self.map, self.DEFAULT_WEIGHT, val_list)

        # update static bomb map
        val_list = [(self.WALL, self.OBSTACLE_WEIGHT),
                      (self.BOX, self.BOX_WEIGHT),
                      (self.ITEM_EXTRA_RANGE, self.BOX_WEIGHT),
                      (self.ITEM_EXTRA_BOMB, self.BOX_WEIGHT)]
        self.static_bomb_map = self.create_map(self.map, self.DEFAULT_WEIGHT, val_list)
        
        for bomb in self.entity_bomb:
            # -2 because we want to predict the timer of the next turn
            reach:tuple = (bomb['reach'], -(bomb['timer']-2))
            obstacle:list = [self.OBSTACLE_WEIGHT]
            val_list:list = [(self.BOX_WEIGHT, -(bomb['timer']-1))]

            self.place_entity(self.static_bomb_map, bomb, reach, obstacle, val_list)

    def place_entity(self, from_map:list, entity:dict, reach:tuple, obstacle:list, affect_val:list) -> None:
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
            # reached obstacle ?
            if from_map[coord[0]][coord[1]] in obstacle:
                return True
            # reached affectable box ?
            elif not direction and __is_affectable(from_map[coord[0]][coord[1]], affect_val):
                from_map[coord[0]][coord[1]] = __get_affected_val(from_map[coord[0]][coord[1]], affect_val)
                return True
            # overwrite only if the cell is empty or if existing timer is greater than the new one (explode sooner)
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
                if entity['y']+i < len(from_map) and not down:
                    down = __check_direction(down, from_map, (entity['y']+i, entity['x']), reach, obstacle, affect_val)
                if entity['x']-i >= 0 and not left:
                    left = __check_direction(left, from_map, (entity['y'], entity['x']-i), reach, obstacle, affect_val)
                if entity['x']+i < len(from_map[0]) and not right:
                    right = __check_direction(right, from_map, (entity['y'], entity['x']+i), reach, obstacle, affect_val)

    def create_map(self, from_map:list, default_val:int, change_val:list) -> list:
        '''
        Description
        ------
            create a new map with weights from the given map

        ----------------
        Param
        ------
            - `from_map`: the map to be converted
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
        new_map = np.full((len(from_map), len(from_map[0])), default_val)

        for i in range(len(from_map)):
            for j in range(len(from_map[0])):
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

class agent:
    '''
    provide player specific information
    
    the following maps are updated:
        - weight map
    '''
    def __init__(self) -> None:
        # constants
        self.BOMB_3:int = 9
        self.BOMB_2:int = 8
        self.BOMB_1:int = 7
        self.SAFE:int = 6
        self.REACHABLE:int = 5

        self.info = information_parser()
        
        # computed map (updated)
        #self.reachable_map:list = []
        self.weight_map:list = []
    
    def update_agent(self) -> None:
        self.info.update_info()

        # apply reachable space
        self.weight_map = np.copy(self.info.static_bomb_map)

        coord:tuple = (self.info.my_player['y'], self.info.my_player['x'])
        obstacle:list = [i for i in range(self.info.OBSTACLE_WEIGHT, 0, 1)] + [self.info.BOX_WEIGHT]
        val_list:list = [(self.info.DEFAULT_WEIGHT, self.REACHABLE)]

        self.spread_map(self.weight_map, coord, obstacle, val_list)

        valid_path = [self.REACHABLE]
        self.temp_coord = self.path_find(self.weight_map, coord, (0, 1), valid_path)

    def spread_map(self, from_map:list, coord:tuple, obstacle:list, change_val:list) -> None:
        '''
        Description
        ------
            spread the map from the given coordinate\n
            convert the chosen value to a given value\n
            can be used to compute the reachable map\n
            it is a recursive function
        
        ----------------
        Param
        ------
            - `from_map`: the map to be spread
            - `coord`: the coordinate to be spread from `tuple(y:int, x:int)`
            - `obstacle`: the obstacle value `list[int, ...]`
            - `change_val`: the values to be changed `list[(from_map value:any, weight:int), ...]`
        ----------------
        Return
        -------
            - `new_map`: the spreaded map
        ----------------
        Raises
        ------
            no raise
        '''
        def __check_direction(from_map:list, coord:tuple, obstacle:list, change_val:list) -> None:
            # check if the coordinate is not an obstacle
            if from_map[coord[0]][coord[1]] not in obstacle:
                # check if the coordinate is already changed
                for pair_val in change_val:
                    if from_map[coord[0]][coord[1]] == pair_val[1]:
                        return
                self.spread_map(from_map, coord, obstacle, change_val)

        # check middle
        for pair_val in change_val:
            if from_map[coord[0]][coord[1]] == pair_val[0]:
                from_map[coord[0]][coord[1]] = pair_val[1]
                break
        # check up
        if coord[0]-1 >= 0:
            __check_direction(from_map, (coord[0]-1, coord[1]), obstacle, change_val)
        # check down
        if coord[0]+1 < len(from_map):
            __check_direction(from_map, (coord[0]+1, coord[1]), obstacle, change_val)
        # check left
        if coord[1]-1 >= 0:
            __check_direction(from_map, (coord[0], coord[1]-1), obstacle, change_val)
        # check right
        if coord[1]+1 < len(from_map[0]):
            __check_direction(from_map, (coord[0], coord[1]+1), obstacle, change_val)

    def path_find(self, from_map:list, coord:tuple, dest:tuple, valid_path:list) -> int:
        '''
        Description
        ------
            Recursive function\n
            search for the shortest path to the destination\n
            when destination is not given, the map will be spreaded to all reachable space
        
        ----------------
        Param
        ------
            - `from_map`: the map to be spread
            - `dim`: the dimension of the map `tuple(height:int, width:int)`
            - `coord`: the coordinate to be spread from `tuple(y:int, x:int)`
            - `dest`: the destination coordinate `tuple(y:int, x:int)`, when not given, `(-1, -1)`
            - `obstacle`: the obstacle value `list[int, ...]`
            - `change_val`: the values to be changed `list[(from_map value:any, weight:int), ...]`
        ----------------
        Return
        -------
            `dist:int`: the shortest distance to the destination (when given)

        ----------------
        Raises
        ------
            in case of invalid destination return -1
        '''
        def __check_direction(from_map:list, visited:list, coord:tuple, valid_path:list) -> bool:
            if coord[0] >= 0 and coord[0] < len(from_map)\
                and coord[1] >= 0 and coord[1] < len(from_map[0])\
                and from_map[coord[0]][coord[1]] in valid_path\
                and not visited[coord[0]][coord[1]]:
                return True
            return False

        def __find_shortest_path(from_map:list, visited:list, coord:tuple, dest:tuple, valid_path:list, min_dist:int, dist:int) -> int:
            if coord[0] == dest[0] and coord[1] == dest[1]:
                min_dist = min(dist, min_dist)
                return min_dist

            visited[coord[0]][coord[1]] = True
            # check up
            up = (coord[0]-1, coord[1])
            if __check_direction(from_map, visited, up, valid_path):
                min_dist = __find_shortest_path(from_map, visited, up, dest, valid_path, min_dist, dist+1)
            # check down
            down = (coord[0]+1, coord[1])
            if __check_direction(from_map, visited, down, valid_path):
                min_dist = __find_shortest_path(from_map, visited, down, dest, valid_path, min_dist, dist+1)
            # check left
            left = (coord[0], coord[1]-1)
            if __check_direction(from_map, visited, left, valid_path):
                min_dist = __find_shortest_path(from_map, visited, left, dest, valid_path, min_dist, dist+1)
            # check right
            right = (coord[0], coord[1]+1)
            if __check_direction(from_map, visited, right, valid_path):
                min_dist = __find_shortest_path(from_map, visited, right, dest, valid_path, min_dist, dist+1)
            
            visited[coord[0]][coord[1]] = False
            return min_dist
  
        if from_map[coord[0]][coord[1]] not in valid_path or from_map[dest[0]][dest[1]] not in valid_path:
            return -1
    
        visited:list = []
        for _ in range(len(from_map)):
            visited.append([None for _ in range(len(from_map[0]))])

        dist:int = sys.maxsize
        dist = __find_shortest_path(from_map, visited, coord, dest, valid_path, dist, 0)

        if dist != sys.maxsize:
            return dist
        return -1

my_agent = agent()

while True:
    my_agent.update_agent()

    #my_agent.info.print_elem('map', info.map)
    #my_agent.info.print_elem('entity_nbr', info.entity_nbr)
    #my_agent.info.print_elem('entity_player', info.entity_player)
    #my_agent.info.print_elem('entity_bomb', info.entity_bomb)
    #my_agent.info.print_elem('entity_item', info.entity_item)
    #my_agent.info.print_elem('my_player', info.my_player)
    #my_agent.info.print_elem('box_map', info.box_map)
    #my_agent.info.print_elem('static_bomb_map', info.static_bomb_map)
    my_agent.info.print_elem('weight_map', my_agent.weight_map)
    my_agent.info.print_elem('test_coord', my_agent.temp_coord)    

    print("MOVE 0 0")