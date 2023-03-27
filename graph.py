from typing import List
import networkx as nx
from dataclasses import dataclass


@dataclass
class Ref:
    title: str
    link: str
    desc: str

    def __hash__(self):
        return hash(self.link)

    def __repr__(self):
        return f"{self.title}\n{self.link}\n{self.desc}"


@dataclass
class MyGraph:
    graph = nx.DiGraph()

    def add_ref(
        self,
        title: str,
        link: str,
        desc: str = None,
        children: List[Ref] = [],
        parents: List[Ref] = [],
    ):
        new_node = Ref(title=title, link=link, desc=desc)
        self.graph.add_node(new_node)
        for child in children:
            self.graph.add_edge(new_node, child)
        for parent in parents:
            self.graph.add_edge(parent, new_node)
        return new_node


G = MyGraph()

# Dropout
dropout = G.add_ref(
    title="Dropout- A Simple Way to Prevent Neural Networks from Overfitting",
    link="www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf",
)

# Maxout Networks
maxout_network = G.add_ref(
    title="Maxout Networks",
    link="arxiv.org/abs/1302.4389",
    desc="Dropout but more",
    parents=[dropout],
)


nx.drawing.nx_pydot.write_dot(G.graph, "test.dot")
