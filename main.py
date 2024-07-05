import networkx as nx


class AgentModel:
    def __init__(self):
        self.__parameters = {"num_nodes": 3, "graph_type": "complete"}
        self.__graph: nx.Graph = None

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

    def initialize_graph(self, initial_data=None) -> nx.Graph:
        num_nodes = self.get_parameter_value("num_nodes")

        if self.graph_type == 'complete':
            self.graph = nx.complete_graph(num_nodes)
        elif self.graph_type == 'cycle':
            self.graph = nx.cycle_graph(num_nodes)
        else:
            self.graph = nx.wheel_graph(num_nodes)

        for node in self.graph.nodes():
            self.graph.nodes[node].update(initial_data)

