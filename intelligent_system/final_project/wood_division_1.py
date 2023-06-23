import sys
import math
import numpy as np

# TODO
# in information_parser class, have a variable to keep in memory the item (unchanged map)
# in strategy class, compute blast radius
# in strategy class, hide unreachable path
# in strategy class, find the closest optimal path
# in agent class, detect when not tp put bombs
# CHECK FOR THE WALLS

class information_parser():
    def __init__(self) -> None:
        self.width, self.height, self.my_id = [int(i) for i in input().split()]

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

class strategy():
    def __init__(self) -> None:
        self.__path = '.'
        self.__wall = 'X'
        self.__box = '0'
        self.__item_range = '1'
        self.__item_bomb = '2'

        self.__unreachable = -9
    
    # create a 2D array of walkable map with 0 for walkable and self.__unreachable for not walkable
    def __compute_walkable_map(self, map:list, height:int, width:int) -> list:
        walkable_map = np.zeros((height, width), dtype=int)
    
        for y, row in enumerate(map):
            for x, col in enumerate(row):
                if not col == self.__path:
                    walkable_map[y][x] = self.__unreachable
        
        return walkable_map

    def __compute_walkable_map_with_unreachable_buffer(self, walkable_map:list, height:int, width:int, x:int, y:int) -> list:
        if y-1 >= 0 and walkable_map[y-1][x] == 0:
            walkable_map[y-1][x] = 1
            self.__compute_walkable_map_with_unreachable_buffer(walkable_map, height, width, x, y-1)
        if y+1 < height and walkable_map[y+1][x] == 0:
            walkable_map[y+1][x] = 1
            self.__compute_walkable_map_with_unreachable_buffer(walkable_map, height, width, x, y+1)
        if x-1 >= 0 and walkable_map[y][x-1] == 0:
            walkable_map[y][x-1] = 1
            self.__compute_walkable_map_with_unreachable_buffer(walkable_map, height, width, x-1, y)
        if x+1 < width and walkable_map[y][x+1] == 0:
            walkable_map[y][x+1] = 1
            self.__compute_walkable_map_with_unreachable_buffer(walkable_map, height, width, x+1, y)
        return walkable_map

    def __compute_walkable_map_with_unreachable(self, walkable_map:list, height:int, width:int, x:int, y:int) -> list:
        walkable_map = self.__compute_walkable_map_with_unreachable_buffer(walkable_map, height, width, x, y)

        for y, row in enumerate(walkable_map):
            for x in range(len(row)):
                if walkable_map[y][x] == 0:
                    walkable_map[y][x] = self.__unreachable
                elif walkable_map[y][x] == 1:
                    walkable_map[y][x] = 0
        return walkable_map

    # TODO avoid blast radius
    # create a 2D array of walkable map with 0 for walkable and self.__unreachable for not walkable
    def __compute_walkable_map_with_bomb(self, walkable_map:list, bomb_list:list) -> list:
        for bomb in bomb_list:
            walkable_map[bomb['y']][bomb['x']] = self.__unreachable
        return walkable_map

    # create a 2D array of box map with location_value+1 for box and items and 0 for not box
    def __compute_box_map(self, map:list, height:int, width:int) -> list:
        box_map = np.zeros((height, width), dtype=int)

        for y, row in enumerate(map):
            for x, col in enumerate(row):
                if not col == self.__path and not col == self.__wall:
                    box_map[y][x] = 1
        
        return box_map

    # compute the score of a location for a bomb
    def __compute_score(self, x:int, y:int, box_map:list, my_player_data:dict) -> int:
        score = 0
        for i in range(1, my_player_data['bomb_reach']):
            if x + i < width:
                score += box_map[y][x + i]
            if x - i >= 0:
                score += box_map[y][x - i]
            if y + i < height:
                score += box_map[y + i][x]
            if y - i >= 0:
                score += box_map[y - i][x]
        return score

    # create a 2D array of weight map with the score of each location for a bomb
    def __compute_weight_map(self, walkable_map:list, box_map:list, my_player_data:dict) -> list:
        for y, row in enumerate(walkable_map):
            for x, col in enumerate(row):
                if col == 0:
                    walkable_map[y][x] = self.__compute_score(x, y, box_map, my_player_data)
        
        return walkable_map

    # return the optimal location for a bomb
    def __get_optimal_bomb_location(self, walkable_map:list) -> tuple:
        max_score = self.__unreachable
        optimal_coord = (0, 0)
        for y, row in enumerate(walkable_map):
            for x, col in enumerate(row):
                if col > max_score:
                    max_score = col
                    optimal_coord = (x, y)
        return optimal_coord

    # return the optimal location for a bomb
    def compute_optimal_bomb_location(self, map:list, height:int, width:int, my_player_data:dict, bomb_list:list) -> tuple:
        box_map = self.__compute_box_map(map, height, width)

        walkable_map = self.__compute_walkable_map(map, height, width)
        walkable_map = self.__compute_walkable_map_with_bomb(walkable_map, bomb_list)
        walkable_map[my_player_data['y']][my_player_data['x']] = 1
        walkable_map = self.__compute_walkable_map_with_unreachable(walkable_map, height, width, my_player_data['x'], my_player_data['y'])

        weight_map = self.__compute_weight_map(walkable_map, box_map, my_player_data)
        
        return self.__get_optimal_bomb_location(weight_map)

class agent():
    def __init__(self) -> None:
        self.strategy = strategy()
    
    # TODO is the optimal location reachable ?
    # idea: just give self.__unreachable value to unreachable location (i guess all location not connected to the player)
    def compute_behaviour(self, map:list, height:int, width:int, my_player_data: dict, bomb_list:list) -> None:
        player_location = (my_player_data['x'], my_player_data['y'])
        location = self.strategy.compute_optimal_bomb_location(map, height, width, my_player_data, bomb_list)

        if my_player_data['nbr_bombs_available'] > 0 and player_location == location:
            self.place_bomb(location[0], location[1])
        else:
            self.move_to(location[0], location[1])

    def move_to(self, x, y):
        print("MOVE",x,y)

    def place_bomb(self, x, y):
        print("BOMB",x,y)

info = information_parser()
agent = agent()

while True:
    info.update_all_info()
    agent.compute_behaviour(info.get_map(), info.get_h(), info.get_w(), info.get_my_player_data(), info.get_entity_bomb())

