import random
import networkx as nx
import numpy as np
from typing import Generator, Callable

MAX_TIMESTEPS = 100000


class AgentModel:
    def __init__(self):
        self.__parameters = {
            "num_nodes": 3,
            "graph_type": "complete",
            "convergence_data_key": None,
            "convergence_std_dev": 100,
        }
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

    def run_to_convergence(self):
        time = 0
        data_key, std_dev = self.get_parameter_value(
            "convergence_data_key"
        ), self.get_parameter_value("convergence_std_dev")

        if not data_key:
            raise Exception("No convergence data key specified")
        while time < MAX_TIMESTEPS and not self.is_converged(data_key, std_dev):
            self.timestep()
            time += 1
        return time

    def is_converged(self, data_key: str, std_dev: float):
        nodes = np.array(
            [node_data[data_key] for _, node_data in self.__graph.nodes(data=True)]
        )
        print(nodes.std())
        return nodes.std() <= std_dev


def genInitialData():
    return {"id": random.randint(1, 100)}


def genTimestepData(nodeData: dict):
    nodeData["id"] = nodeData["id"] + 1
    return nodeData
