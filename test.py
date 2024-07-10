import unittest
from main import AgentModel, genInitialData


class testAgentModelPackage(unittest.TestCase):
    def testInit(self):
        agent = AgentModel()
        self.assertIsInstance(agent, AgentModel)

    def testParameterMethods(self):
        agent = AgentModel()
        self.assertListEqual(
            ["num_nodes", "graph_type", "convergence_data_key", "convergence_std_dev"],
            agent.list_parameters(),
        )
        
    def testInitGraph(self):
        agent = AgentModel()
        agent.set_initial_data_function(genInitialData)
        agent.initialize_graph()
        graph_nodes = list(agent.get_graph().nodes)
        self.assertEqual(graph_nodes, [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
