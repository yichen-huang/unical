import sys
import math
import numpy as np

# TODO
# item map
# bomb map
# predict bomb radius
# predict map state after bomb explosion


class information_parser():
    def __init__(self) -> None:
        self.map_w, self.map_h, self.my_id = [int(i) for i in input().split()]

        self.map = []
        self.entity_nbr = 0

        self.entity_player = []
        self.entity_bomb = []
        self.entity_item = []
        self.my_player = {}

    def __set_map(self) -> list:
        map = []

        for i in range(self.map_h):
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

    def get_map(self) -> list:
        return self.map

    def get_w(self) -> int:
        return self.map_w

    def get_h(self) -> int:
        return self.map_h

    def get_id(self) -> int:
        return self.my_id

    def get_entity_number(self) -> int:
        return self.entity_nbr
    
    def get_entity_player(self) -> list:
        return self.entity_player

    def get_entity_bomb(self) -> list:
        return self.entity_bomb

    def get_my_player_data(self) -> dict:
        return self.my_player

    def print_map(self) -> None:
        for i in range(height):
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

    def print_all_entities(self) -> None:
        print("entity number", self.entity_nbr, file=sys.stderr, flush=True)
        self.print_entity_player()
        self.print_entity_bomb()
        self.print_my_player_data()
        self.print_entity_item()

# TODO
# compute bomb map
# compute item map
class map_compute():
    def __init__(self, height:int, width:int) -> None:
        self.__height = height
        self.__width = width

        self.__path = '.'
        self.__wall = 'X'
        self.__box = '0'
        self.__item_range = '1'
        self.__item_bomb = '2'

        self.__unreachable = -9
        self.__wall_weight = -1

        self.walkable_map = []
        self.box_map = []
        self.weight_map = []
    
    # create a 2D array
    # 0 for walkable
    # self.__unreachable for not walkable
    def compute_walkable_map(self, map:list) -> None:
        self.walkable_map = np.zeros((self.__height, self.__width), dtype=int)
    
        for y, row in enumerate(map):
            for x, col in enumerate(row):
                if not col == self.__path:
                    self.walkable_map[y][x] = self.__unreachable

    # uses the walkable map
    # set to 1 all the walkable cells around the player
    def __apply_walkable_map_with_unreachable_buffer(self,x:int, y:int) -> list:
        if y-1 >= 0 and self.walkable_map[y-1][x] == 0:
            self.walkable_map[y-1][x] = 1
            self.__apply_walkable_map_with_unreachable_buffer(x, y-1)
        if y+1 < self.__height and self.walkable_map[y+1][x] == 0:
            self.walkable_map[y+1][x] = 1
            self.__apply_walkable_map_with_unreachable_buffer(x, y+1)
        if x-1 >= 0 and self.walkable_map[y][x-1] == 0:
            self.walkable_map[y][x-1] = 1
            self.__apply_walkable_map_with_unreachable_buffer(x-1, y)
        if x+1 < self.__width and self.walkable_map[y][x+1] == 0:
            self.walkable_map[y][x+1] = 1
            self.__apply_walkable_map_with_unreachable_buffer(x+1, y)
        return self.walkable_map

    # uses the walkable map
    # set to 1 all the walkable cells around the player
    # set to self.__unreachable all zero values cells (meaning not reachable by the player)
    # reset to 0 all the 1 values cells (meaning reachable by the player)
    def apply_walkable_map_with_unreachable(self, my_player_data:dict) -> None:
        x = my_player_data['x']
        y = my_player_data['y']
        self.walkable_map = self.__apply_walkable_map_with_unreachable_buffer(x, y)

        for y, row in enumerate(self.walkable_map):
            for x in range(len(row)):
                if self.walkable_map[y][x] == 0:
                    self.walkable_map[y][x] = self.__unreachable
                elif self.walkable_map[y][x] == 1:
                    self.walkable_map[y][x] = 0

    # TODO predict blast radius
    # uses the walkable map
    # set to self.__unreachable all the cells with a bomb
    def apply_walkable_map_with_bomb(self, bomb_list:list) -> None:
        for bomb in bomb_list:
            self.walkable_map[bomb['y']][bomb['x']] = self.__unreachable

    # create a 2D array
    # set to 1 all the box cells (regardless of items)
    # set to self.__wall_weight all the wall cells
    def compute_box_map(self, map:list) -> None:
        self.box_map = np.zeros((self.__height, self.__width), dtype=int)

        for y, row in enumerate(map):
            for x, col in enumerate(row):
                if not col == self.__path and not col == self.__wall:
                    self.box_map[y][x] = 1
                elif col == self.__wall:
                    self.box_map[y][x] = self.__wall_weight

    # uses the box map
    # check each direction for a box and only one box
    # don't check for a box if a wall is in the way
    # returns an int value of the number of boxes a bomb can destroy
    def __apply_score(self, x:int, y:int, my_player_data:dict) -> int:
        up = False
        down = False
        left = False
        right = False
        score = 0

        for i in range(1, my_player_data['bomb_reach']):
            if x + i < self.__width and not right and (self.box_map[y][x + i] == 1 or self.box_map[y][x + i] == self.__wall_weight):
                score += self.box_map[y][x + i] if self.box_map[y][x + i] == 1 else 0
                right = True
            if x - i >= 0 and not left and (self.box_map[y][x - i] == 1 or self.box_map[y][x - i] == self.__wall_weight):
                score += self.box_map[y][x - i] if self.box_map[y][x - i] == 1 else 0
                left = True
            if y + i < self.__height and not down and (self.box_map[y + i][x] == 1 or self.box_map[y + i][x] == self.__wall_weight):
                score += self.box_map[y + i][x] if self.box_map[y + i][x] == 1 else 0
                down = True
            if y - i >= 0 and not up and (self.box_map[y - i][x] == 1 or self.box_map[y - i][x] == self.__wall_weight):
                score += self.box_map[y - i][x] if self.box_map[y - i][x] == 1 else 0
                up = True
        return score

    # create a 2D array
    # uses the walkable map
    # uses the box map
    # set each reachable cell with a score of how many boxes a bomb can destroy
    def compute_weight_map(self, my_player_data:dict) -> None:
        self.weight_map = np.copy(self.walkable_map)

        for y, row in enumerate(self.walkable_map):
            for x, col in enumerate(row):
                if col == 0:
                    self.weight_map[y][x] = self.__apply_score(x, y, my_player_data)

    # update walkable map
    # update box map
    # update weight map
    def compute_all_maps(self, map:list, my_player_data:dict, bomb_list:list) -> None:
        self.compute_box_map(map)

        self.compute_walkable_map(map)
        self.apply_walkable_map_with_bomb(bomb_list)
        self.walkable_map[my_player_data['y']][my_player_data['x']] = 1
        self.apply_walkable_map_with_unreachable(my_player_data)

        self.compute_weight_map(my_player_data)

    # return the optimal location for a bomb
    def get_optimal_bomb_location(self) -> tuple:
        max_score = self.__unreachable
        optimal_coord = (0, 0)
        for y, row in enumerate(self.weight_map):
            for x, col in enumerate(row):
                if col > max_score:
                    max_score = col
                    optimal_coord = (x, y)
        return optimal_coord

    def get_walkable_map(self) -> list:
        return self.walkable_map
    
    def get_box_map(self) -> list:
        return self.box_map
    
    def get_weight_map(self) -> list:
        return self.weight_map
    
    def print_walkable_map(self) -> None:
        print("walkable_map\n", self.walkable_map, file=sys.stderr, flush=True)

    def print_box_map(self) -> None:
        print("box_map\n", self.box_map, file=sys.stderr, flush=True)

    def print_weight_map(self) -> None:
        print("weight_map\n", self.weight_map, file=sys.stderr, flush=True)
    
    def print_all_maps(self) -> None:
        self.print_walkable_map()
        self.print_box_map()
        self.print_weight_map()

# TODO
# predict bomb radius
# predict map state after bomb explosion
class simulation():
    def __init__(self) -> None:
        pass

class agent():
    def __init__(self) -> None:
        pass

    def compute_behaviour(self, my_player_data: dict, optimal_location: tuple) -> None:
        player_location = (my_player_data['x'], my_player_data['y'])
        location = optimal_location

        if my_player_data['nbr_bombs_available'] > 0 and player_location == location:
            self.place_bomb(location[0], location[1])
        else:
            self.move_to(location[0], location[1])

    def move_to(self, x, y):
        print("MOVE",x,y)

    def place_bomb(self, x, y):
        print("BOMB",x,y)

info = information_parser()
cpu = map_compute(info.get_h(), info.get_w())
agent = agent()

while True:
    info.update_all_info()

    cpu.compute_all_maps(info.get_map(), info.get_my_player_data(), info.get_entity_bomb())

    agent.compute_behaviour(info.get_my_player_data(), cpu.get_optimal_bomb_location())
