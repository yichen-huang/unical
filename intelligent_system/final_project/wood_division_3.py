import sys
import math
import numpy as np

class information_getters():
    def __init__(self, map_w: int, map_h: int, my_id: int) -> None:
        self.map_w = map_w
        self.map_h = map_h
        self.my_id = my_id

        self.map = []
        self.entity_nbr = 0

        self.entity_player = []
        self.entity_bomb = []
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
            else:
                entity_player.append({
                    'type': entity_type,
                    'owner': owner,
                    'x': x,
                    'y': y,
                    'nbr_bombs_available': param_1,
                    'bomb_reach': param_2
                    })
        return entity_player, entity_bomb

    def __set_my_player(self) -> dict:
        for player in self.entity_player:
            if player['owner'] == self.my_id:
                return player
        return None

    def update_all_info(self) -> None:
        self.map = self.__set_map()
        self.entity_nbr = self.__set_entity_number()
        self.entity_player, self.entity_bomb = self.__set_info_each_entity()
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

class strategy():
    def __init__(self) -> None:
        self.__path = '.'
        self.__box = '0'
    
    # create a 2D array of walkable map with 0 for walkable and -1000 for not walkable
    def compute_walkable_map(self, map:list, height:int, width:int) -> list:
        walkable_map = np.zeros((height, width), dtype=int)

        for y, row in enumerate(map):
            for x, col in enumerate(row):
                if not col == self.__path:
                    walkable_map[y][x] = -1000
        
        return walkable_map
    
    # create a 2D array of box map with 1 for box and 0 for not box
    def compute_box_map(self, map:list, height:int, width:int) -> list:
        box_map = np.zeros((height, width), dtype=int)

        for y, row in enumerate(map):
            for x, col in enumerate(row):
                if col == self.__box:
                    box_map[y][x] = 1
        
        return box_map

    # create a 2D array of weight map with the score of each location for a bomb
    def compute_weight_map(self, walkable_map:list, box_map:list, my_player_data:dict) -> list:
        weight_map = np.copy(walkable_map)

        for y, row in enumerate(walkable_map):
            for x, col in enumerate(row):
                if col == 0:
                    weight_map[y][x] = self.compute_score(x, y, box_map, my_player_data)
        
        return weight_map

    # compute the score of a location for a bomb
    def compute_score(self, x:int, y:int, box_map:list, my_player_data:dict) -> int:
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

    # return the optimal location for a bomb
    def get_optimal_bomb_location(self, walkable_map:list) -> tuple:
        max_score = -1000
        optimal_coord = (0, 0)
        for y, row in enumerate(walkable_map):
            for x, col in enumerate(row):
                if col > max_score:
                    max_score = col
                    optimal_coord = (x, y)
        return optimal_coord

    # return the optimal location for a bomb
    def compute_optimal_bomb_location(self, map:list, height:int, width:int, my_player_data:dict) -> tuple:
        walkable_map = self.compute_walkable_map(map, height, width)
        box_map = self.compute_box_map(map, height, width)
        weight_map = self.compute_weight_map(walkable_map, box_map, my_player_data)
        
        return self.get_optimal_bomb_location(weight_map)

class agent():
    def __init__(self) -> None:
        self.strategy = strategy()
    
    def compute_behaviour(self, map:list, height:int, width:int, my_player_data) -> None:
        player_location = (my_player_data['x'], my_player_data['y'])
        location = self.strategy.compute_best_location(map, height, width, my_player_data)

        if my_player_data['nbr_bombs_available'] > 0 and player_location == location:
            self.place_bomb(location[0], location[1])
        else:
            self.move_to(location[0], location[1])

    def move_to(self, x, y):
        print("MOVE",x,y)

    def place_bomb(self, x, y):
        print("BOMB",x,y)

width, height, my_id = [int(i) for i in input().split()]

info = information_getters(width, height, my_id)
#strat = strategy()
agent = agent()

while True:
    info.update_all_info()

    #info.print_map()
    #info.print_all_entities()
    #print("walk_map\n",strat.compute_walkable_map(info.get_map(), info.get_h(), info.get_w()), file=sys.stderr, flush=True)
    #print("box_map\n",strat.compute_box_map(info.get_map(), info.get_h(), info.get_w()), file=sys.stderr, flush=True)
    #print("weight_map\n",strat.compute_weight_map(
    #    strat.compute_walkable_map(info.get_map(), info.get_h(), info.get_w()),
    #    strat.compute_box_map(info.get_map(), info.get_h(),info.get_w()),
    #    info.get_my_player_data()),
    #    file=sys.stderr, flush=True)
    #print("optimal_location\n",strat.compute_optimal_bomb_location(
    #    info.get_map(),
    #    info.get_h(),
    #    info.get_w(),
    #    info.get_my_player_data()),
    #    file=sys.stderr, flush=True)
    agent.compute_behaviour(info.get_map(), info.get_h(), info.get_w(), info.get_my_player_data())

