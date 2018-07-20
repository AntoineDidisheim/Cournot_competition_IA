import numpy as np


class CournotCompetitionGame:
    def __init__(self, number_player, type_history = "simple"):
        assert type_history in ["comp_1", "comp_2","simple"], "type_history must be 'comp_*' or 'simple"

        self.number_player = number_player
        self.type_history = type_history
        self.history_quantity = np.empty([number_player])
        self.history_costs = np.empty([number_player])
        self.history_price = np.empty([1,1])
        self.history_profits = np.empty([number_player])
        self.expected_shape = self.history_quantity.shape
        self.max_price = 100
        self.quantity_slope = 1
        self.cost_slope = 0.5
        self.first_entry = True # simple work around to know when removing the first row of history

    def get_competitive_equilibrium_production(self):
        q = 0
        if self.number_player == 2:
            q = 2*(self.max_price-self.cost_slope)/3
        return q

    def get_all_history(self):

        if self.type_history == "simple":
            # columns: price, quantity, profits,costs (last three time number_player)
            if len(self.history_price)>1:
                out = np.hstack([
                    # self.history_price.reshape(len(self.history_price),1),
                    #              self.history_quantity,
                    #              self.history_profits,
                                 self.history_costs])
            else:
                out = np.hstack([
                    # self.history_price.reshape((1,1)),
                    #              self.history_quantity.reshape(1,self.number_player),
                    #              self.history_profits.reshape(1,self.number_player),
                                 self.history_costs.reshape(1,self.number_player)])
            out = out*0
        elif self.type_history in ["comp_2","comp_1"]:
            if len(self.history_price)>1:
                out = np.hstack([self.history_quantity])
                                 # , self.history_profits])
            else:
                out = np.hstack([self.history_quantity.reshape(1,self.number_player)])
                                 # , self.history_profits.reshape(1,self.number_player)])


        return out

    def get_lag_history(self,lag = 1):
        if self.type_history == "simple":
            lag_history = self.get_all_history()[-lag,:]
        elif self.type_history == "comp_1":
            lag_history = self.get_all_history()[-lag, :]
        elif self.type_history == "comp_2":
            if len(self.history_price)>1: #this means we have more than one history stored
                lag_history = np.append(self.get_all_history()[-lag, :],
                                        self.get_all_history()[-(lag+1), :])
            else:
                lag_history = np.append(self.get_all_history()[-lag, :],
                                        self.get_all_history()[-(lag), :])

        return lag_history

    def get_price(self, produced_quantities):
        assert type(produced_quantities) == np.ndarray, "produced_quantities must be a np.ndarray"
        assert produced_quantities.shape == self.expected_shape, "shape of produced_quantities does match"

        price = self.max_price-self.quantity_slope*np.sum(produced_quantities)
        return price

    def get_costs(self, produced_quantities):
        assert type(produced_quantities) == np.ndarray, "produced_quantities must be a np.ndarray"
        assert produced_quantities.shape == self.expected_shape, "shape of produced_quantities does match"

        costs = produced_quantities*self.cost_slope
        return costs

    def get_profits(self, produced_quantities, consult_only=False):
        assert type(produced_quantities) == np.ndarray, "produced_quantities must be a np.ndarray"
        assert produced_quantities.shape == self.expected_shape, "shape of produced_quantities does match"

        costs = self.get_costs(produced_quantities)
        price = self.get_price(produced_quantities)

        profits = price*produced_quantities-costs

        # saving in the history all prices costs quantities and profits
        if not consult_only:
            self.history_quantity = np.vstack([self.history_quantity, produced_quantities])
            self.history_profits = np.vstack([self.history_profits, profits])
            self.history_costs = np.vstack([self.history_costs, costs])
            self.history_price = np.append(self.history_price, price)
            if self.first_entry:
                # if its first_entry we remove the first rows
                self.first_entry = False
                self.history_quantity = self.history_quantity[1:, ]
                self.history_profits = self.history_profits[1:, ]
                self.history_costs = self.history_costs[1:, ]
                self.history_price = self.history_price[1:]


        return profits

