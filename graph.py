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
    desc="Max of layer after masking; better gradient flow for deeper networks; maxout unit linear almost everywhere",
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
# Conv 
kaiming_init_conv = G.add_ref(
    title="Delving Deep into Rectifiers- Surpassing Human-Level Performance on ImageNet Classification",
    link="arxiv.org/abs/1502.01852",
    desc="Kaiming Initialization",
    parents=[cs231n]
)

resnet = G.add_ref(
    title="Deep Residual Learning for Image Recognition",
    link="arxiv.org/abs/1512.03385",
    parents=[cs231n]
)

efficient_net = G.add_ref(
    title="EfficientNet- Rethinking Model Scaling for Convolutional Neural Networks",
    link="arxiv.org/pdf/1905.11946.pdf",
    desc="How to scale conv nets",
    parents=[resnet],
)

efficient_net_v2 = G.add_ref(
    title="EfficientNetV2- Smaller Models and Faster Training",
    link="arxiv.org/abs/2104.00298",
    desc="Efficient net but better?",
    parents=[efficient_net],
)

squeeze_net = G.add_ref(
    title="SqueezeNet- AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size",
    link="arxiv.org/abs/1602.07360",
    desc="Super small conv nets",
    parents=[efficient_net],
)

dense_net = G.add_ref(
    title="Densely Connected Convolutional Networks",
    link="arxiv.org/abs/1608.06993",
    parents=[resnet]
)

highway_net = G.add_ref(
    title="Highway Networks",
    link="arxiv.org/abs/1505.00387",
    parents=[resnet]
)

saliency_maps = G.add_ref(
    title="Deep Inside Convolutional Networks- Visualising Image Classification Models and Saliency Maps",
    link="arxiv.org/abs/1312.6034",
    desc="Saliency Maps",
    parents=[cs231n]
)

fooling_iamges = G.add_ref(
    title="Intriguing properties of neural networks",
    link="arxiv.org/abs/1312.6199",
    desc="Fooling images",
    parents=[saliency_maps]
)
#####################################################################################################################
# RNNs
training_RNNs = G.add_ref(
    title="TRAINING RECURRENT NEURAL NETWORKS",
    link="www.cs.utoronto.ca/~ilya/pubs/ilya_sutskever_phd_thesis.pdf",
    desc="Ilya Sutskever's dissertation",
    parents=[cs231n]
)

#####################################################################################################################
# NLP 
neural_machine_translation = G.add_ref(
    title="Sequence to Sequence Learning with Neural Networks",
    link="arxiv.org/pdf/1409.3215.pdf",
    desc="Neural machine translation/ Encoder-decoder arch",
)

#####################################################################################################################
# Attention
bahdanau_attention = G.add_ref(
    title="NEURAL MACHINE TRANSLATION BY JOINTLY LEARNING TO ALIGN AND TRANSLATE",
    link="arxiv.org/pdf/1409.0473.pdf",
    desc="First paper to introduce attention mechanism",
    parents=[neural_machine_translation],
)   

luong_attention = G.add_ref(
    title="Effective Approaches to Attention-based Neural Machine Translation",
    link="arxiv.org/pdf/1508.04025.pdf",
    desc="Multiplicative Attention",
    parents=[bahdanau_attention],
)   

self_attention = G.add_ref(
    title="Long Short-Term Memory-Networks for Machine Reading",
    link="arxiv.org/pdf/1601.06733.pdf",
    desc="Self-Attention Introduction",
    parents=[bahdanau_attention],
)   

image_captioning_attention = G.add_ref(
    title="Show, Attend and Tell- Neural Image Caption Generation with Visual Attention",
    link="arxiv.org/pdf/1502.03044.pdf",
    desc="Image captioning with attention",
    parents=[bahdanau_attention],
)   

#####################################################################################################################

# Self-Instruct
self_instruct = G.add_ref(
    title="Self-Instruct- Aligning Language Model with Self Generated Instructions",
    link="arxiv.org/abs/2212.10560",
)
# Alpaca
alpaca = G.add_ref(
    title="Alpaca- A Strong, Replicable Instruction-Following Model",
    link="crfm.stanford.edu/2023/03/13/alpaca.html",
    desc="Fine turning llama w/ GPT",
    parents=[self_instruct],
)


#####################################################################################################################
# Self-supervised learning

ssl_cookbook = G.add_ref(
    title="A Cookbook of Self-Supervised Learning",
    link="arxiv.org/pdf/2304.12210.pdf",
    desc="Good stuff",
)

sim_clr = G.add_ref(
    title="A Simple Framework for Contrastive Learning of Visual Representations",
    link="arxiv.org/pdf/2002.05709.pdf",
    desc="Self Supervised Learning pretraining for images",
    parents=[cs231n,ssl_cookbook],
)

#####################################################################################################################
# Distributed Stuff
large_scale_distributed_deep_networks = G.add_ref(
    title="Large Scale Distributed Deep Networks",
    link="papers.nips.cc/paper/2012/file/6aca97005c68f1206823815f66102863-Paper.pdf",
)       

revisiting_distributed_sgd = G.add_ref(
    title="REVISITING DISTRIBUTED SYNCHRONOUS SGD",
    link="arxiv.org/pdf/1604.00981.pdf",
    desc="clever idea- drop slowest b stragglers. poisson dist has thin tails.",
    parents=[large_scale_distributed_deep_networks],
)   

#####################################################################################################################
# Geometric Deep Learning
geometric_deep_learning = G.add_ref(title="Geometric Deep Learning- Grids, Groups, Graphs, Geodesics, and Gauges", link="arxiv.org/abs/2104.13478")

nx.drawing.nx_pydot.write_dot(G.graph, "test.dot")
