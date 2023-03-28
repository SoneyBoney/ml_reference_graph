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

# CS231n
cs231n = G.add_ref(title="CS231n", link="cs231n.github.io/", desc="Solid introduction.")

# Dropout
dropout = G.add_ref(
    title="Dropout- A Simple Way to Prevent Neural Networks from Overfitting",
    link="www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf",
    parents=[cs231n],
    desc="Big regularization."
)

# Maxout Networks
maxout_network = G.add_ref(
    title="Maxout Networks",
    link="arxiv.org/abs/1302.4389",
    desc="Dropout but more",
    parents=[dropout],
)

# Batch Norm
batch_norm = G.add_ref(
    title="Batch Normalization- Accelerating Deep Network Training by Reducing Internal Covariate Shift",
    link="arxiv.org/abs/1502.03167",
    parents=[cs231n],
    desc="Stabilizes training, more robust to initial conditions"
)

# Layer Norm
layer_norm = G.add_ref(
    title="Layer Normalization", link="arxiv.org/abs/1607.06450", parents=[batch_norm],
    desc="Batchnorm sucks when batchsize is small; enter layer norm"
)

# Group Norm
group_norm = G.add_ref(
    title="Group Normalization", link="arxiv.org/abs/1803.08494", parents=[batch_norm]
)

#####################################################################################################################

# Alpaca
alpaca = G.add_ref(
    title="Alpaca- A Strong, Replicable Instruction-Following Model",
    link="crfm.stanford.edu/2023/03/13/alpaca.html",
    desc="Fine turning llama w/ GPT"
)

# Self-Instruct
self_instruct = G.add_ref(
    title="Self-Instruct- Aligning Language Model with Self Generated Instructions",
    link="arxiv.org/abs/2212.10560",
    parents=[alpaca],
)

#####################################################################################################################

# Geometric Deep Learning
geometric_deep_learning = G.add_ref(title="Geometric Deep Learning- Grids, Groups, Graphs, Geodesics, and Gauges", link="arxiv.org/abs/2104.13478")

nx.drawing.nx_pydot.write_dot(G.graph, "test.dot")
