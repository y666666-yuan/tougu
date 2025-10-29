NODE_LABEL_INVESTOR = '投资者'
NODE_LABEL_PRODUCT = '产品'


class BaseNetworkNode:

    def __init__(self, id: str, label: str, properties):
        self.id = id
        self.label = label
        self.properties = properties

    @staticmethod
    def build_investor(id, properties):
        return BaseNetworkNode(id, NODE_LABEL_INVESTOR, properties)

    @staticmethod
    def build_product(id, properties):
        return BaseNetworkNode(id, NODE_LABEL_PRODUCT, properties)


class BaseNetworkEdge:

    def __init__(self, id: str, label: str, edge_from: str, edge_to: str, value=None):
        self.id = id
        self.label = label
        self.edge_from = edge_from
        self.edge_to = edge_to
        self.value = value


class BaseNetwork:

    def __init__(self, nodes: list, edges: list):
        self.nodes = nodes
        self.edges = edges
