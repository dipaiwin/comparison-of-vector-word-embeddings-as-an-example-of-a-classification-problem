from tabulate import tabulate

from config_project import ConfigProject
from network.network import ManagerNetwork
from network.node_create_manager import NodeCreateManager


class NNManager:
    def __init__(self, config: ConfigProject, path):
        self.cfg = config
        self.path = path
        self.values = NodeCreateManager(self.cfg).create_nodes()
        ManagerNetwork(self.values, self.cfg).create_network()
    
    def run(self):
        raise NotImplementedError()
    
    def culc_statistic(self, correct_answer, net_answer, origin_ids):
        statistic = dict()
        for net, cor, o_i in zip(net_answer, correct_answer, origin_ids):
            if cor not in statistic:
                statistic[cor] = { 'sum': 0, 'correct': 0, 'origin_id': o_i }
            statistic[cor]['sum'] += 1
            if cor == net:
                statistic[cor]['correct'] += 1
        for key in statistic.keys():
            statistic[key]['acc'] = statistic[key]['correct'] / statistic[key]['sum']
        self.print_table(statistic)
    
    def print_table(self, statistic):
        table = [
            [
                'Class:{0}'.format(key),
                'Origin_id:{0}'.format(statistic[key]['origin_id']),
                statistic[key]['acc']
            ] for key in statistic
        ]
        print(tabulate(table))
