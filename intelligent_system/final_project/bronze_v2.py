import math
from time import process_time
import sys
import numpy as np

class information_parser():
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
        self.t1_start = process_time() 
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
        
        # update maps
        val_list:tuple = [(self.WALL, self.OBSTACLE_WEIGHT),
                      (self.BOX, self.BOX_WEIGHT),
                      (self.ITEM_EXTRA_RANGE, self.BOX_WEIGHT),
                      (self.ITEM_EXTRA_BOMB, self.BOX_WEIGHT)]
        
        # update box map
        self.box_map = self.create_map(self.map, self.DEFAULT_WEIGHT, val_list)
        # update static bomb map
        self.static_bomb_map = self.create_map(self.map, self.DEFAULT_WEIGHT, val_list)
        
        for bomb in self.entity_bomb:
            # -2 because we want to predict the timer of the next turn
            reach:tuple = (bomb['reach'], -(bomb['timer']-1))
            obstacle:list = [self.OBSTACLE_WEIGHT]
            val_list:list = [(self.BOX_WEIGHT, -(bomb['timer']))]
            
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
    def __init__(self) -> None:
        # constants
        self.SAFE:int = 10
        self.BOMB_4:int = 9
        self.BOMB_3:int = 8
        self.BOMB_2:int = 7
        self.BOMB_1:int = 6
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

        self.spread_map(self.weight_map, coord, obstacle, val_list, self.info.entity_bomb, True, 10)

        # apply weights
        valid_path:list = [self.REACHABLE]
        obstacle = [self.info.OBSTACLE_WEIGHT]
        target:list = [self.info.BOX_WEIGHT]

        # update weights and get the best location
        best_location = self.compute_weight(self.weight_map, self.info.my_player, valid_path, obstacle, target)

        # gives commands
        if self.weight_map[best_location[0]][best_location[1]] in [self.BOMB_1, self.BOMB_2, self.BOMB_3, self.BOMB_4]\
            and best_location == (self.info.my_player['y'], self.info.my_player['x']):
            print("BOMB", best_location[1], best_location[0])
        else:
            print("MOVE", best_location[1], best_location[0])

    def spread_map(self, from_map:list, coord:tuple, obstacle:list, change_val:list, entity_bomb:list, ignore_first_blast:bool, depth:int) -> None:
        '''
        Param
        ------
            - `from_map`: the map to be spread
            - `coord`: the coordinate to be spread from `tuple(y:int, x:int)`
            - `obstacle`: the obstacle value `list[int, ...]`
            - `change_val`: the values to be changed `list[(from_map value:any, weight:int), ...]`
            - `entity_bomb`: list of the bombs `list[dict{}, ...]`
            - `ignore_first_blast`: ignore the first blast of the bomb `bool`, in case the player is in the blast range
            - `depth`: the depth of the spread `int`
        '''
        def __check_direction(from_map:list, prev_coord:tuple, coord:tuple, obstacle:list, change_val:list, entity_bomb:list, ignore_first_blast:bool, depth:int) -> None:
            # check if the coordinate is not a bomb
            for bomb in entity_bomb:
                if coord[0] == bomb['y'] and coord[1] == bomb['x']:
                    return
            # check if the coordinate can ignore the first blast
            if ignore_first_blast and from_map[coord[0]][coord[1]] in obstacle\
            and self.info.box_map[coord[0]][coord[1]] not in [self.info.OBSTACLE_WEIGHT, self.info.BOX_WEIGHT]\
            and from_map[coord[0]][coord[1]] == from_map[prev_coord[0]][prev_coord[1]]:
                self.spread_map(from_map, coord, obstacle, change_val, entity_bomb, ignore_first_blast, depth-1)
            # check if the coordinate is not an obstacle
            if from_map[coord[0]][coord[1]] not in obstacle:
                # check if the coordinate is already changed
                for pair_val in change_val:
                    if from_map[coord[0]][coord[1]] == pair_val[1]:
                        return
                self.spread_map(from_map, coord, obstacle, change_val, entity_bomb, False, depth-1)

        if depth == 0:
            return
        # check middle
        for pair_val in change_val:
            if from_map[coord[0]][coord[1]] == pair_val[0]:
                from_map[coord[0]][coord[1]] = pair_val[1]
                break
        # check up
        if coord[0]-1 >= 0:
            __check_direction(from_map,(coord[0], coord[1]), (coord[0]-1, coord[1]), obstacle, change_val, entity_bomb, ignore_first_blast, depth)
        # check down
        if coord[0]+1 < len(from_map):
            __check_direction(from_map,(coord[0], coord[1]), (coord[0]+1, coord[1]), obstacle, change_val, entity_bomb, ignore_first_blast, depth)
        # check left
        if coord[1]-1 >= 0:
            __check_direction(from_map,(coord[0], coord[1]), (coord[0], coord[1]-1), obstacle, change_val, entity_bomb, ignore_first_blast, depth)
        # check right
        if coord[1]+1 < len(from_map[0]):
            __check_direction(from_map,(coord[0], coord[1]), (coord[0], coord[1]+1), obstacle, change_val, entity_bomb, ignore_first_blast, depth)

    def compute_weight(self, from_map:list, my_player:dict, valid_path:list, obstacle:list, target:list) -> tuple:
        '''
        Param
        ------
            - `from_map`: the map to computed
            - `my_player`: the player information `dict {...'x':int, 'y':int, 'bomb':int, 'range':int ...}`
            - `valid_path`: reachable value `list[int, ...]`
            - `obstacle`: value that block us `list[int, ...]`
            - `target`: value we want to destroy `list[int, ...]`
        '''
        def __is_in_elem(from_map:list, coord:tuple, elem:list) -> bool:
            if coord[0] >= 0 and coord[0] < len(from_map)\
                and coord[1] >= 0 and coord[1] < len(from_map[0])\
                and from_map[coord[0]][coord[1]] in elem:
                    return True
            return False

        def __count_entity(from_map:list, coord:tuple, obstacle:list, target:list, reach:int) -> int:
            count:int = 0

            up = False
            down = False
            left = False
            right = False
            for i in range(1, reach):
                if not up and __is_in_elem(from_map, (coord[0]-i, coord[1]), obstacle + target):
                    up = True
                    count += 1 if __is_in_elem(from_map, (coord[0]-i, coord[1]), target) else 0
                if not down and __is_in_elem(from_map, (coord[0]+i, coord[1]), obstacle + target):
                    down = True
                    count += 1 if __is_in_elem(from_map, (coord[0]+i, coord[1]), target) else 0
                if not left and __is_in_elem(from_map, (coord[0], coord[1]-i), obstacle + target):
                    left = True
                    count += 1 if __is_in_elem(from_map, (coord[0], coord[1]-i), target) else 0
                if not right and __is_in_elem(from_map, (coord[0], coord[1]+i), obstacle + target):
                    right = True
                    count += 1 if __is_in_elem(from_map, (coord[0], coord[1]+i), target) else 0
            return count

        def __find_best_closest(from_map:list, my_player, valid_path:list) -> tuple:
            best_score:int = -999
            min_dist:int = sys.maxsize
            best_location:tuple = (-1, -1)

            for i in range(len(from_map)):
                for j in range(len(from_map[0])):
                    cur_dist:int = math.dist((my_player['y'], my_player['x']), (i, j))
                    if from_map[i][j] in valid_path and from_map[i][j] >= best_score:
                        if from_map[i][j] == best_score and cur_dist < min_dist or from_map[i][j] > best_score:
                            best_score = from_map[i][j]
                            min_dist = cur_dist
                            best_location = (i, j)
            #self.info.print_elem('final map', from_map)
            #self.info.print_elem('min_dist', min_dist)
            #self.info.print_elem('best_score', best_score)
            #self.info.print_elem('best_location', best_location)
            #self.info.print_elem('final best location', best_location)
            return best_location

        def __simulate_bomb(from_map:list, my_player:dict, best_location:tuple, obstacle:list, target:list) -> list:
            if best_location == (-1, -1):
                return from_map

            new_map:list = np.copy(from_map)
            location:dict = {'y':best_location[0], 'x':best_location[1]}
            reach:tuple = (my_player['reach'], -(8-2))
            val_list:list = [(i, -(8-1)) for i in target]

            self.info.place_entity(new_map, location, reach, obstacle, val_list)

            return new_map

        def __is_safe(simulated_map:list, my_player:dict, valid_path:list) -> bool:
            #no_bomb_left:bool = True
            simulated_safe:bool = False
            
            for i in range(len(simulated_map)):
                for j in range(len(simulated_map[0])):
                    if simulated_map[i][j] in valid_path:
                        if math.dist((my_player['y'], my_player['x']), (i, j)) <= 8-2:
                            simulated_safe = True
            return simulated_safe

        # initialize the weights
        for i in range(len(from_map)):
            for j in range(len(from_map[0])):
                if from_map[i][j] in valid_path:
                    from_map[i][j] += __count_entity(from_map, (i, j), obstacle, target, my_player['reach'])

        # remove the unsafe locations
        is_safe:bool = False
        valid_path = [self.BOMB_1, self.BOMB_2, self.BOMB_3, self.BOMB_4]
        best_location:tuple = __find_best_closest(from_map, my_player, valid_path)
        simulated_map:list = __simulate_bomb(from_map, my_player, best_location, obstacle, target)

        while not is_safe:
            if not __is_safe(simulated_map, my_player, valid_path+[self.REACHABLE]):
                from_map[best_location[0]][best_location[1]] = self.REACHABLE
                best_location = __find_best_closest(from_map, my_player, valid_path)
                simulated_map = __simulate_bomb(from_map, my_player, best_location, obstacle, target)
            else:
                is_safe = True
            #self.info.print_elem('weight_map', from_map)
            #self.info.print_elem('simulated_map', simulated_map)
        
        # remove all other weights
        best_location:tuple = __find_best_closest(from_map, my_player, valid_path+[self.REACHABLE])
        for i in range(len(from_map)):
            for j in range(len(from_map[0])):
                if from_map[i][j] in valid_path and (i, j) != best_location:
                    from_map[i][j] = self.REACHABLE
        #self.info.print_elem('final map', from_map)
        return best_location

my_agent = agent()

while True:
    my_agent.update_agent()
    #my_agent.info.print_elem('weight_map', my_agent.weight_map)
