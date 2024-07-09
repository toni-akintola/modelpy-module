import random
import networkx as nx
from typing import Generator, Callable


class AgentModel:
    def __init__(self):
        self.__parameters = {"num_nodes": 3, "graph_type": "complete"}
        self.__graph: nx.Graph = None
        self.initial_data_generator = None
        self.timestep_generator = None

    def update_parameters(self, parameters: dict) -> None:
        self.__parameters.update(parameters)

    def delete_parameters(self, parameters: list) -> None:
        for param in parameters:
            if param not in self.__parameters or param in {"num_nodes", "graph_type"}:
                raise KeyError
            self.__parameters.pop(param)
        return True

    def list_parameters(self, parameters: list = None) -> list:
        if not parameters:
            return list(self.__parameters.keys())

        result = []

        for param in parameters:
            if param not in self.__parameters:
                raise KeyError
            result.append(param)

        return result

    def get_parameter_value(self, parameter: str) -> None | float | int | str:
        if self.__parameters.get(parameter):
            return self.__parameters[parameter]
        raise KeyError

    def get_graph(self):
        return self.__graph

    def set_initial_data_generator(self, initial_data_generator: Callable):
        self.initial_data_generator = initial_data_generator

    def set_timestep_data_generator(self, timestep_generator: Callable):
        self.timestep_generator = timestep_generator

    def initialize_graph(self):
        num_nodes = self.get_parameter_value("num_nodes")
        graph_type = self.get_parameter_value("graph_type")

        if graph_type == "complete":
            self.__graph = nx.complete_graph(num_nodes)
        elif graph_type == "cycle":
            self.__graph = nx.cycle_graph(num_nodes)
        else:
            self.__graph = nx.wheel_graph(num_nodes)

        for node in self.__graph.nodes():
            initial_data = self.initial_data_generator()
            self.__graph.nodes[node].update(initial_data)

    def timestep(self):
        for _node, node_data in self.__graph.nodes(data=True):
            node_data = self.timestep_generator(node_data)


test = AgentModel()


def genInitialData():
    return {"id": random.randint(1, 100)}


def genTimestepData(nodeData: dict):
    nodeData["id"] = nodeData["id"] + 1
    return nodeData


test.set_initial_data_generator(genInitialData)
test.set_timestep_data_generator(genTimestepData)

test.initialize_graph()

graph = test.get_graph()

print(graph.nodes(data=True))
test.timestep()
print(graph.nodes(data=True))
