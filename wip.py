from cournot import CournotCompetitionGame
import numpy as np
import random
from player import PlayerRndForrest
from player import PlayerNNet
nb_player = 10
# creating the game with some fixed number of player
game = CournotCompetitionGame(nb_player,type_history="comp_1")

players = []
for i in range(nb_player):
    pl = PlayerNNet(game=game, end_of_random_action_time=1500, action_range=10,random_action_range=1)
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

    if s % 100 ==0:
        print('---- current progress is:',100*round(s/S,3),'% ----')

game.get_all_history()
game.get_lag_history()
game.history_profits
game.history_quantity

from matplotlib import pyplot as plt

temp = np.array(game.history_quantity.tolist())
plt.plot(game.history_quantity)
plt.title("Quantity")
plt.show()

plt.plot(game.history_profits)
plt.title("profits")
plt.show()

if len(players)>1:
    # total quantity porduced
    plt.plot(np.sum(game.history_quantity,1))
    plt.title("total quantity produced")
    plt.show()

    # total quantity porduced
    plt.plot(np.sum(game.history_profits,1))
    plt.title("aggregated profits")
    plt.show()


np.mean(np.sum(game.history_quantity,1))
from scipy.stats import ttest_1samp
qty=np.sum(game.history_quantity,1)
qty = qty[-500:]
print(np.mean(qty))
print(ttest_1samp(qty,popmean=game.get_competitive_equilibrium_production()))
