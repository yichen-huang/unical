from math import inf
import sys
import numpy as np

# TODO
# item map

class information_parser():
    """
    class information_parser

    parse all the information given by the game

    the following information are parsed:
        - map
        - map width
        - map height
        - player id
        - number of entities
        - player entities
        - bomb entities
        - item entities

    the following information are updated:
        - map
        - number of entities
        - player entities
        - bomb entities
        - item entities
    """
    def __init__(self) -> None:
        self.width, self.height, self.my_id = [int(i) for i in input().split()]

        self.map:list = []
        self.entity_nbr:int = 0

        self.entity_player:list = []
        self.entity_bomb:list = []
        self.entity_item:list = []

        self.my_player:dict = {}

    def __set_map(self) -> list:
        map = []

        for _ in range(self.height):
            row = input()
            map.append(row)
        return map

    def __set_entity_number(self) -> int:
        return int(input())

    def __set_info_each_entity(self) -> list:
        entity_bomb = []
        entity_player = []
        entity_item = []

        for i in range(self.entity_nbr):
            entity_type, owner, x, y, param_1, param_2 = [int(j) for j in input().split()]
            if entity_type == 1:
                entity_bomb.append({
                    'type': entity_type,
                    'owner': owner,
                    'x': x,
                    'y': y,
                    'timer': param_1,
                    'bomb_reach': param_2
                    })
            elif entity_type == 0:
                entity_player.append({
                    'type': entity_type,
                    'owner': owner,
                    'x': x,
                    'y': y,
                    'nbr_bombs_available': param_1,
                    'bomb_reach': param_2
                    })
            elif entity_type == 2:
                entity_item.append({
                    'type': entity_type,
                    'x': x,
                    'y': y,
                    'item_type': param_1
                    })
        return entity_player, entity_bomb, entity_item

    def __set_my_player(self) -> dict:
        for player in self.entity_player:
            if player['owner'] == self.my_id:
                return player
        return None

    def update_all_info(self) -> None:
        self.map = self.__set_map()
        self.entity_nbr = self.__set_entity_number()
        self.entity_player, self.entity_bomb, self.entity_item = self.__set_info_each_entity()
        self.my_player = self.__set_my_player()

    def print_map(self) -> None:
        for i in range(self.height):
            print("row", self.map[i], file=sys.stderr, flush=True)

    def print_entity_player(self) -> None:
        for player in self.entity_player:
            print("player", player['owner'],
                    'at', player['x'], player['y'],
                    ', bombs available and reach',
                    player['nbr_bombs_available'],
                    player['bomb_reach'],
                    file=sys.stderr, flush=True)

    def print_entity_bomb(self) -> None:
        for bomb in self.entity_bomb:
            print("bomb", bomb['owner'],
                    'at', bomb['x'], bomb['y'],
                    ', timer and reach', bomb['timer'], bomb['bomb_reach'],
                    file=sys.stderr, flush=True)

    def print_entity_item(self) -> None:
        for item in self.entity_item:
            print("item", item['x'], item['y'], item['item_type'],
                    file=sys.stderr, flush=True)

    def print_my_player_data(self) -> None:
        print("my_player", self.my_player['owner'],
                'at', self.my_player['x'], self.my_player['y'],
                ', bombs available and reach',
                self.my_player['nbr_bombs_available'], self.my_player['bomb_reach'],
                file=sys.stderr, flush=True)

# TODO
# compute item map
class map_compute():
    """
    class map_compute

    aggregate all information available from `information_parser()` class and compute them into maps.
    all maps are 2D array of int.

    list of maps:
        - walkable map: 0 for walkable by the player, -9 for unreachable
        - box map: 0 for no box, 1 for box, -1 for wall
        - dynamic bomb map: map with the bomb radius of each cell, the radius changes according to the timer
        - hard bomb map: map with the bomb radius of each cell, the radius is constant
        - item map: TBA

    note: the default value for each map is 0
    """
    def __init__(self, height:int, width:int) -> None:
        self.__height = height
        self.__width = width

        self.__path = '.'
        self.__wall = 'X'
        self.__box = '0'
        self.__item_extra_range = '1'
        self.__item_extra_bomb = '2'

        # every map
        # can be walls, boxes, bombs
        self.__obstacle_weight = -9

        # box map
        self.__box_weight = 1

        # bomb map
        self.__bomb_radius_weight = -5

        self.item_map = []

        self.box_map = []
        self.dynamic_bomb_map = []
        self.timed_dynamic_bomb_map = []
        self.hard_bomb_map = []

        self.walkable_map = []
        self.safe_spot_map = []
        self.hard_safe_spot_map = []
    
    # uses the walkable map
    # set to 1 all the walkable cells around the player
    def __apply_walkable_map_with_unreachable_buffer(self, walkable_map, x:int, y:int) -> None:
        if y-1 >= 0 and walkable_map[y-1][x] == 0:
            walkable_map[y-1][x] = 1
            self.__apply_walkable_map_with_unreachable_buffer(walkable_map, x, y-1)
        if y+1 < self.__height and walkable_map[y+1][x] == 0:
            walkable_map[y+1][x] = 1
            self.__apply_walkable_map_with_unreachable_buffer(walkable_map, x, y+1)
        if x-1 >= 0 and walkable_map[y][x-1] == 0:
            walkable_map[y][x-1] = 1
            self.__apply_walkable_map_with_unreachable_buffer(walkable_map, x-1, y)
        if x+1 < self.__width and walkable_map[y][x+1] == 0:
            walkable_map[y][x+1] = 1
            self.__apply_walkable_map_with_unreachable_buffer(walkable_map, x+1, y)

    # uses the walkable map
    # set to 1 all the walkable cells around the player
    # set to self.__obstacle_weight all zero values cells (meaning not reachable by the player)
    # reset to 0 all the 1 values cells (meaning reachable by the player)
    def __apply_walkable_map_with_unreachable(self, walkable_map:list, my_player:dict) -> None:
        x = my_player['x']
        y = my_player['y']
        walkable_map[y][x] = 1

        self.__apply_walkable_map_with_unreachable_buffer(walkable_map, x, y)
        for y, row in enumerate(walkable_map):
            for x in range(len(row)):
                if walkable_map[y][x] == 0:
                    walkable_map[y][x] = self.__obstacle_weight
                elif walkable_map[y][x] == 1:
                    walkable_map[y][x] = 0

    # uses the walkable map
    # set to self.__obstacle_weight all the cells with a bomb
    def __apply_walkable_map_with_bomb(self, walkable_map:list, bomb_list:list) -> None:
        for bomb in bomb_list:
            walkable_map[bomb['y']][bomb['x']] = self.__obstacle_weight

    # create a 2D array
    # 0 for walkable
    # self.__obstacle_weight for not walkable
    def compute_walkable_map(self, map:list, my_player:dict, bomb_list:list) -> list:
        walkable_map = np.zeros((self.__height, self.__width), dtype=int)
    
        for y, row in enumerate(map):
            for x, col in enumerate(row):
                if not col == self.__path:
                    walkable_map[y][x] = self.__obstacle_weight

        self.__apply_walkable_map_with_bomb(walkable_map, bomb_list)
        self.__apply_walkable_map_with_unreachable(walkable_map, my_player)

        # have to cover this case independently because apply unreachable overwrite the player position
        for bomb in bomb_list:
            if bomb['x'] == my_player['x'] and bomb['y'] == my_player['y']:
                walkable_map[bomb['y']][bomb['x']] = self.__obstacle_weight
        
        return walkable_map

    # create a 2D array
    # set to 1 all the box cells (regardless of items)
    # set to self.__obstacle_weight all the wall cells
    def compute_box_map(self, map:list) -> list:
        box_map = np.zeros((self.__height, self.__width), dtype=int)

        for y, row in enumerate(map):
            for x, col in enumerate(row):
                if not col == self.__path and not col == self.__wall:
                    box_map[y][x] = self.__box_weight
                elif col == self.__wall:
                    box_map[y][x] = self.__obstacle_weight
        
        return box_map

    # uses any bomb map
    # set to self.__bomb_radius_weight all the cells in the bomb radius according to the computed range
    # /!\ DOES NOT check if the bomb is in the radius of another bomb
    def __apply_bomb_map_radius(self, bomb_map:list, bomb:dict, computed_range:int, value:int) -> None:
        up = False
        down = False
        left = False
        right = False
        
        # set the center bomb cell to the bomb radius weight
        bomb_map[bomb['y']][bomb['x']] = value
        for i in range(1, computed_range):
            # if conditions to avoid out computating range through boxes and walls,
            # because the radius stops at the first obstacle
            if bomb['y']-i >= 0 and bomb_map[bomb['y']-i][bomb['x']] == self.__obstacle_weight:
                up = True
            if bomb['y']+i < self.__height and bomb_map[bomb['y']+i][bomb['x']] == self.__obstacle_weight:
                down = True
            if bomb['x']-i >= 0 and bomb_map[bomb['y']][bomb['x']-i] == self.__obstacle_weight:
                left = True
            if bomb['x']+i < self.__width and bomb_map[bomb['y']][bomb['x']+i] == self.__obstacle_weight:
                right = True

            # compute the bomb radius according to the computed range
            if bomb['y']-i >= 0 and bomb_map[bomb['y']-i][bomb['x']] != self.__obstacle_weight and not up:
                bomb_map[bomb['y']-i][bomb['x']] = value
            if bomb['y']+i < self.__height and bomb_map[bomb['y']+i][bomb['x']] != self.__obstacle_weight and not down:
                bomb_map[bomb['y']+i][bomb['x']] = value
            if bomb['x']-i >= 0 and bomb_map[bomb['y']][bomb['x']-i] != self.__obstacle_weight and not left:
                bomb_map[bomb['y']][bomb['x']-i] = value
            if bomb['x']+i < self.__width and bomb_map[bomb['y']][bomb['x']+i] != self.__obstacle_weight and not right:
                bomb_map[bomb['y']][bomb['x']+i] = value

    # create a 2D array
    # uses the box map
    # set to self.__obstacle_weight all the box cells and the wall cells
    # set to self.__bomb_radius_weight all the cells in the bomb radius according to the timer
    def compute_dynamic_bomb_map(self, box_map:list, bomb_list:list) -> list:
        dynamic_bomb_map = np.copy(box_map)

        for y, row in enumerate(dynamic_bomb_map):
            for x, col in enumerate(row):
                if col == self.__box_weight:
                    dynamic_bomb_map[y][x] = self.__obstacle_weight

        for bomb in bomb_list:
            # -2 because we want to predict the bomb radius of the next turn
            # -1 for the current turn radius
            computed_range = min(bomb['bomb_reach'] - min(bomb['timer'] - 2, bomb['bomb_reach']), bomb['bomb_reach'])
            self.__apply_bomb_map_radius(dynamic_bomb_map, bomb, computed_range, self.__bomb_radius_weight)
        
        return dynamic_bomb_map

    def compute_timed_dynamic_bomb_map(self, box_map:list, bomb_list:list) -> list:
        timed_dynamic_bomb_map = np.copy(box_map)

        for y, row in enumerate(timed_dynamic_bomb_map):
            for x, col in enumerate(row):
                if col == self.__box_weight:
                    timed_dynamic_bomb_map[y][x] = self.__obstacle_weight

        for bomb in bomb_list:
            # -2 because we want to predict the bomb radius of the next turn
            # -1 for the current turn radius
            computed_range = min(bomb['bomb_reach'] - min(bomb['timer'] - 2, bomb['bomb_reach']), bomb['bomb_reach'])
            self.__apply_bomb_map_radius(timed_dynamic_bomb_map, bomb, computed_range, -(bomb['timer']-1))
        
        return timed_dynamic_bomb_map

    # create a 2D array
    # uses the box map
    # set to self.__obstacle_weight all the box cells and the wall cells
    # set to self.__bomb_radius_weight all the cells in the bomb radius
    def compute_hard_bomb_map(self, box_map:list, bomb_list:list) -> list:
        hard_bomb_map = np.copy(box_map)

        for y, row in enumerate(hard_bomb_map):
            for x, col in enumerate(row):
                if col == self.__box_weight:
                    hard_bomb_map[y][x] = self.__obstacle_weight

        for bomb in bomb_list:
            self.__apply_bomb_map_radius(hard_bomb_map, bomb, bomb['bomb_reach'], self.__bomb_radius_weight)
        
        return hard_bomb_map

    # create a 2D array
    # uses the walkable map
    # uses a bomb map
    # set to self.__obstacle_weight all the cells in the bomb radius
    def compute_safe_spot_map(self, walkable_map:list, bomb_map:list, my_player) -> list:
        safe_spot_map = np.copy(walkable_map)

        for y, row in enumerate(safe_spot_map):
            for x, _ in enumerate(row):
                if bomb_map[y][x] == self.__bomb_radius_weight:
                    safe_spot_map[y][x] = self.__obstacle_weight

        self.__apply_walkable_map_with_unreachable(safe_spot_map, my_player)

        # have to cover this case independently because apply unreachable overwrite the player position
        if bomb_map[my_player['y']][my_player['x']] == self.__bomb_radius_weight:
            safe_spot_map[my_player['y']][my_player['x']] = self.__obstacle_weight

        return safe_spot_map

    # update box map
    # update dynamic bomb map
    # update hard bomb map
    # update walkable map
    # update safe spot map
    def compute_all_maps(self, map:list, my_player:dict, bomb_list:list) -> None:
        self.box_map = self.compute_box_map(map)
        self.dynamic_bomb_map = self.compute_dynamic_bomb_map(self.box_map, bomb_list)
        self.timed_dynamic_bomb_map = self.compute_timed_dynamic_bomb_map(self.box_map, bomb_list)
        self.hard_bomb_map = self.compute_hard_bomb_map(self.box_map, bomb_list)

        self.walkable_map = self.compute_walkable_map(map, my_player, bomb_list)
        self.safe_spot_map = self.compute_safe_spot_map(self.walkable_map, self.dynamic_bomb_map, my_player)
        self.hard_safe_spot_map = self.compute_safe_spot_map(self.walkable_map, self.hard_bomb_map, my_player)

    def print_walkable_map(self) -> None:
        print("walkable_map\n", self.walkable_map, file=sys.stderr, flush=True)

    def print_box_map(self) -> None:
        print("box_map\n", self.box_map, file=sys.stderr, flush=True)

    def print_dynamic_bomb_map(self) -> None:
        print("dynamic_bomb_map\n", self.dynamic_bomb_map, file=sys.stderr, flush=True)
   
    def print_timed_dynamic_bomb_map(self) -> None:
        print("timed_dynamic_bomb_map\n", self.timed_dynamic_bomb_map, file=sys.stderr, flush=True)

    def print_hard_bomb_map(self) -> None:
        print("hard_bomb_map\n", self.hard_bomb_map, file=sys.stderr, flush=True)

    def print_safe_spot_map(self) -> None:
        print("safe_spot_map\n", self.safe_spot_map, file=sys.stderr, flush=True)

    def print_hard_safe_spot_map(self) -> None:
        print("hard_safe_spot_map\n", self.hard_safe_spot_map, file=sys.stderr, flush=True)

class simulation():
    """
    class simulation

    simulate the game state after:
        - a bomb explosion
        - a bomb placement
    simulate the game state after a bomb placement or explosion:
        - best move after bomb placement
        - is the placement safe

        - weight map: map with the weight of each cell (higher is better)
    """
    def __init__(self, height:int, width:int) -> None:
        self.__height = height
        self.__width = width

        # every map
        self.__obstacle_weight = -9

        # box map
        self.__box_weight = 1

        # bomb map
        self.__bomb_radius_weight = -5

        # weight map
        self.__proximity_weight_bonus = 5
        self.__proximity_weight_malus = -5

        self.weight_map = []
        self.bomb_placement_map = []

    # apply on the ring (EDGE ONLY around the player in a square shape)
    # a proximity score to the weight map
    # the closer the cell is to the player, the higher the score
    def __apply_proximity_score_ring(self, weight_map:list, my_player:dict, ring:int, proximity_weight:int, bias:int) -> None:
        # upside and downside will avoid the corners (smaller range)
        # the corners are already covered by the left and right side
        # this will avoid to count twice the corners
        for i in range(-ring+1, ring):
            # upside
            y = my_player['y'] - ring
            x = my_player['x'] + i
            if x >= 0 and x < self.__width and y >= 0:
                if weight_map[y][x] > 0:
                    weight_map[y][x] *= bias
                    weight_map[y][x] += proximity_weight

            # downside
            y = my_player['y'] + ring
            x = my_player['x'] + i
            if x >= 0 and x < self.__width and y < self.__height:
                if weight_map[y][x] > 0:
                    weight_map[y][x] *= bias
                    weight_map[y][x] += proximity_weight
        
        # leftside and rightside will count the corners
        for i in range(-ring, ring+1):
            # leftside
            y = my_player['y'] + i
            x = my_player['x'] - ring
            if y >= 0 and y < self.__height and x >= 0:
                if weight_map[y][x] > 0:
                    weight_map[y][x] *= bias
                    weight_map[y][x] += proximity_weight
            
            # rightside
            y = my_player['y'] + i
            x = my_player['x'] + ring
            if y >= 0 and y < self.__height and x < self.__width:
                if weight_map[y][x] > 0:
                    weight_map[y][x] *= bias
                    weight_map[y][x] += proximity_weight

    # apply on the player and around the player
    # a proximity score to the weight map
    def __apply_proximity_score_bonus(self, weight_map:list, my_player:dict) -> None:
        y = my_player['y']
        x = my_player['x']

        if weight_map[y][x] > 0:
            weight_map[y][x] *= 2
            weight_map[y][x] += self.__proximity_weight_bonus
        for i in range(1, 5):
            self.__apply_proximity_score_ring(weight_map, my_player, i, self.__proximity_weight_bonus-i, 2)

    # apply around the bomb
    # a proximity score to the weight map
    # avoid putting a bomb next to a bomb
    def __apply_proximity_score_malus(self, weight_map:list, hard_bomb_map:list) -> None:
        for y, row in enumerate(hard_bomb_map):
            for x, _ in enumerate(row):
                if hard_bomb_map[y][x] == self.__bomb_radius_weight and weight_map[y][x] >= 0:
                    weight_map[y][x] += self.__proximity_weight_malus
                    weight_map[y][x] = max(weight_map[y][x], 0)

    # uses the box map
    # check each direction for a box and only one box
    # don't check for a box if a wall is in the way
    # returns an int value of the number of boxes a bomb can destroy
    def __apply_score(self, x:int, y:int, my_player:dict, box_map:list) -> int:
        up = False
        down = False
        left = False
        right = False
        score = 0

        for i in range(1, my_player['bomb_reach']):
            if x + i < self.__width and not right \
            and (box_map[y][x + i] == self.__box_weight or box_map[y][x + i] == self.__obstacle_weight):
                score += box_map[y][x + i] if box_map[y][x + i] == self.__box_weight else 0
                right = True
            if x - i >= 0 and not left \
            and (box_map[y][x - i] == self.__box_weight or box_map[y][x - i] == self.__obstacle_weight):
                score += box_map[y][x - i] if box_map[y][x - i] == self.__box_weight else 0
                left = True
            if y + i < self.__height and not down \
            and (box_map[y + i][x] == self.__box_weight or box_map[y + i][x] == self.__obstacle_weight):
                score += box_map[y + i][x] if box_map[y + i][x] == self.__box_weight else 0
                down = True
            if y - i >= 0 and not up \
            and (box_map[y - i][x] == self.__box_weight or box_map[y - i][x] == self.__obstacle_weight):
                score += box_map[y - i][x] if box_map[y - i][x] == self.__box_weight else 0
                up = True
        return score

    # create a 2D array
    # uses the walkable map
    # uses the box map
    # uses the hard bomb map
    # set each reachable cell with a score of how many boxes a bomb can destroy
    def simulate_weight_map(self, my_player:dict, safe_spot_map:list, box_map:list, hard_bomb_map:list) -> list:
        weight_map = np.copy(safe_spot_map)

        for y, row in enumerate(safe_spot_map):
            for x, col in enumerate(row):
                if col == 0:
                    weight_map[y][x] = self.__apply_score(x, y, my_player, box_map)
        
        self.__apply_proximity_score_bonus(weight_map, my_player)
        self.__apply_proximity_score_malus(weight_map, hard_bomb_map)

        return weight_map

    # create a 2D array
    # uses the walkable map
    # simulate a walkable map with bomb placement
    def simulate_bomb_placement_map(self, x:int, y:int, my_player:dict, safe_spot_map:list) -> list:
        bomb_placement_map = np.copy(safe_spot_map)
    
        bomb_placement_map[y][x] = self.__bomb_radius_weight
        for i in range(1, my_player['bomb_reach']):
            if x + i < self.__width and safe_spot_map[y][x + i] != self.__obstacle_weight:
                bomb_placement_map[y][x + i] = self.__bomb_radius_weight
            if x - i >= 0 and safe_spot_map[y][x - i] != self.__obstacle_weight:
                bomb_placement_map[y][x - i] = self.__bomb_radius_weight
            if y + i < self.__height and safe_spot_map[y + i][x] != self.__obstacle_weight:
                bomb_placement_map[y + i][x] = self.__bomb_radius_weight
            if y - i >= 0 and safe_spot_map[y - i][x] != self.__obstacle_weight:
                bomb_placement_map[y - i][x] = self.__bomb_radius_weight
        
        return bomb_placement_map
    
    # check the bomb_placement_map if there is any safe spot left
    # avoid self kill situation
    def __safe_spot_available(self, bomb_placement_map) -> bool:
        for row in bomb_placement_map:
            for col in row:
                if col == 0:
                    return True
        return False
    
    # check if there are safe zone
    # are the safe zone in any blast radius
    def simulate_safe_zone_map(self, my_player:dict, hard_safe_spot_map:list, weight_map:list) -> list:
        for y, row in enumerate(weight_map):
            for x, _ in enumerate(row):
                if weight_map[y][x] > 0 :
                    self.bomb_placement_map = self.simulate_bomb_placement_map(x, y, my_player, hard_safe_spot_map)
                    if not self.__safe_spot_available(self.bomb_placement_map):
                        weight_map[y][x] = 0

        return weight_map

    # calculate the weight map
    def simulate(self, my_player:dict, safe_spot_map:list, hard_safe_spot_map, box_map:list, hard_bomb_map:list) -> None:
        self.weight_map = self.simulate_weight_map(my_player, safe_spot_map, box_map, hard_bomb_map)
        self.weight_map = self.simulate_safe_zone_map(my_player, hard_safe_spot_map, self.weight_map)

    # return the optimal location for a bomb
    def get_optimal_bomb_location(self) -> tuple:
        max_score = self.__obstacle_weight
        optimal_coord = (0, 0)
        for y, row in enumerate(self.weight_map):
            for x, col in enumerate(row):
                if col > max_score:
                    max_score = col
                    optimal_coord = (x, y)
        return optimal_coord
    
    def print_weight_map(self) -> None:
        print("weight_map\n", self.weight_map, file=sys.stderr, flush=True)

class agent():
    """
    class agent

    compute the best behaviour for the agent with all the information given
    """
    def __init__(self, height, width) -> None:
        self.__height = height
        self.__width = width

    def __find_immediate_safe_spot(self, my_player:dict, timed_dynamic_bomb_map:list) -> tuple:
        y = my_player['y']
        x = my_player['x']

        if timed_dynamic_bomb_map[y][x] == 0:
            return (x, y)
        if y + 1 < self.__height and timed_dynamic_bomb_map[y + 1][x] == 0:
            return (x, y + 1)
        if y - 1 >= 0 and timed_dynamic_bomb_map[y - 1][x] == 0:
            return (x, y - 1)
        if x + 1 < self.__width and timed_dynamic_bomb_map[y][x + 1] == 0:
            return (x + 1, y)
        if x - 1 >= 0 and timed_dynamic_bomb_map[y][x - 1] == 0:
            return (x - 1, y)
        return (-1, -1)

    def __any_incoming_explosion(self, my_player:dict, timed_dynamic_bomb_map:list) -> bool:
        y = my_player['y']
        x = my_player['x']

        if timed_dynamic_bomb_map[y][x] == -2:
            return True
        if y + 1 < self.__height and timed_dynamic_bomb_map[y + 1][x] == -1:
            return True
        if y - 1 >= 0 and timed_dynamic_bomb_map[y - 1][x] == -1:
            return True
        if x + 1 < self.__width and timed_dynamic_bomb_map[y][x + 1] == -1:
            return True
        if x - 1 >= 0 and timed_dynamic_bomb_map[y][x - 1] == -1:
            return True
        return False

    def compute_behaviour(self, my_player: dict, optimal_location:tuple, timed_dynamic_bomb_map:list, weight_map:list) -> None:
        player_location = (my_player['x'], my_player['y'])
        location = optimal_location

        if self.__any_incoming_explosion(my_player, timed_dynamic_bomb_map):
            location = self.__find_immediate_safe_spot(my_player, timed_dynamic_bomb_map)
            self.move_to(location[0], location[1])
        elif my_player['nbr_bombs_available'] > 0 and player_location == location:
            if weight_map[location[1]][location[0]] > 0:
                self.place_bomb(location[0], location[1])
            else:
                self.move_to(location[0], location[1])
        else:
            self.move_to(location[0], location[1])

    def move_to(self, x, y):
        print("MOVE",x,y)

    def place_bomb(self, x, y):
        print("BOMB",x,y)

info = information_parser()
cpu = map_compute(info.height, info.width)
simulator = simulation(info.height, info.width)
my_agent = agent(info.height, info.width)

while True:
    info.update_all_info()

    cpu.compute_all_maps(info.map, info.my_player, info.entity_bomb)

    #cpu.print_box_map()
    #cpu.print_hard_bomb_map()
    #cpu.print_dynamic_bomb_map()
    cpu.print_timed_dynamic_bomb_map()
    #cpu.print_walkable_map()
    #cpu.print_safe_spot_map()
    cpu.print_hard_safe_spot_map()

    simulator.simulate(info.my_player, cpu.safe_spot_map, cpu.hard_safe_spot_map, cpu.box_map, cpu.hard_bomb_map)

    simulator.print_weight_map()
    #print("bomb_placement_map\n", simulator.bomb_placement_map, file=sys.stderr, flush=True)

    my_agent.compute_behaviour(info.my_player, simulator.get_optimal_bomb_location(), cpu.timed_dynamic_bomb_map, simulator.weight_map)
