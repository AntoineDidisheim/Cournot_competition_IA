from cournot import CournotCompetitionGame
import numpy as np
import random
from player import PlayerRndForrest
import pickle
from player import PlayerNNet
from matplotlib import pyplot as plt
nb_player = 2
# creating the game with some fixed number of player

if True:
    games = []
    for meta_sim in range(1000):
        game = CournotCompetitionGame(nb_player, type_history="comp_2")
        games.append(game)
        players = []
        for i in range(nb_player):
            pl = PlayerNNet(game=game, end_of_random_action_time=1500, action_range=10,random_action_range=10)
            pl.production=random.randint(10,70)
            players.append(pl)
            del pl

        S = 2000 # number of round in simulation
        for s in range(S):
            state = game.get_lag_history() # we save here the current state use for decision making
            for pl in players: # loop through all palyer to define their action
                pl_action = pl.select_action(current_state=state, step=s)
                pl.production = max(0,pl.production+pl_action)

            production_list = []
            for temp_pl in players:
                production_list.append(temp_pl.production)
            production_vector = np.array(production_list)
            del production_list
            # compute profits this round and update the game
            profits = game.get_profits(produced_quantities=production_vector)
            next_state = game.get_lag_history()  # we save here the next state that is going to be used for next round
            for j in range(len(players)): # loop again now to update the q function
                pl = players[j]
                pl.update_q(state=state, next_state=next_state, reward=profits[j])

            # if s % 100 ==0:
            #     print('---- current progress is:',100*round(s/S,3),'% ----')
        print('Meta_sim ', meta_sim, 'done')
        if meta_sim%10==0:
            with open('pickle/sim_'+str(meta_sim)+'.pickle', 'wb') as handle:
                pickle.dump(games, handle, protocol=pickle.HIGHEST_PROTOCOL)


games = pickle.load(open("pickle/sim_90.pickle", "rb" ))

mean_qty = []
for game in games:
    qty=np.sum(game.history_quantity,1)
    qty = qty[-500:]
    mean_qty.append(np.mean(qty))
from matplotlib import pyplot as plt
plt.hist(mean_qty,100)
plt.show()
# prft=np.sum(game.history_profits,1)
# prft = prft[-500:]

