import unittest
from main import AgentModel


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


if __name__ == "__main__":
    unittest.main()
