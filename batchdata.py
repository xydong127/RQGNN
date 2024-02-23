
class Batch:
    def __init__(self, adj_list, features_list, label_list, graphpool_list, lap_list, edge_index, label_count, node_belong, xLx_batch):
        self.adj_list = adj_list
        self.features_list = features_list
        self.label_list = label_list
        self.graphpool_list = graphpool_list
        self.lap_list = lap_list
        self.edge_index = edge_index
        self.label_count = label_count
        self.node_belong = node_belong
        self.xLx_batch = xLx_batch
