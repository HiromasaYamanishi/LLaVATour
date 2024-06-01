import torch
import torch.nn as nn
import numpy as np
from sklearn.neighbors import BallTree#KDTree
from torch_geometric.nn import GCNConv

class PositionalGeoEmbedding(nn.Module):
    def __init__(self, coordinates):
        self.position_embedding = self._calc_position_embeddings(coordinates)
        
    def _calc_position_embedding(coords, dim=5120):
        normalized_coords = torch.zeros_like(coords)
        normalized_coords[:, 0] = (coords[:, 0] + 90) / 180  # 緯度
        normalized_coords[:, 1] = (coords[:, 1] + 180) / 360  # 経度

        # 周波数をスケーリングするためのパラメータ
        frequencies = torch.linspace(0.1, 10, dim // 2)  # dim/2 frequencies

        # 角度を計算 (N, 2, dim/2)
        angles = 2 * np.pi * normalized_coords[:, :, None] * frequencies[None, None, :]

        # sinとcosの計算 (N, 2, dim)
        embeddings = torch.zeros(coords.shape[0], 2, dim)
        embeddings[:, :, ::2] = torch.sin(angles)
        embeddings[:, :, 1::2] = torch.cos(angles)

        return embeddings
    
    def forward(self, index):
        return self.position_embedding[index]
    
class GraphGeoModule(nn.Module):
    def __init__(self, coordinates, k=10, max_dist=100.0, hidden_dim=5120):
        super().__init__()
        self.original_coordinates = coordinates
        self.k = k
        self.max_dist = max_dist
        #print('coordinates', coordinates)
        # 無効な座標をフィルタリング (-1, -1)は無効な座標と仮定
        valid_indices = [i for i, coord in enumerate(coordinates) if coord[0] != -1 and coord[1] != -1]
        valid_coordinates = [coordinates[i] for i in valid_indices]

        # 座標をラジアンに変換してKDTreeを構築
        self.num_nodes = len(self.original_coordinates)
        radian_coordinates = np.radians(valid_coordinates)
        self.tree = BallTree(radian_coordinates, metric='haversine')

        # エッジのリストを作成
        distances, indices = self.tree.query(radian_coordinates, k=min(k+1, self.num_nodes), return_distance=True)
        self.edge_index = self._create_edge(distances, indices, valid_indices)

        # 学習可能な地点特性
        self.node_features = nn.Parameter(torch.randn(self.num_nodes, hidden_dim//16))
        
        # GNN Layer
        self.conv1 = GCNConv(hidden_dim//16, hidden_dim//4)
        self.conv2 = GCNConv(hidden_dim//4, hidden_dim)

    def _create_edge(self, distances, indices, valid_indices):
        # エッジを作成するメソッドを修正して、実際のノードインデックスを考慮
        edges = []
        for i in range(len(distances)):
            for j in range(1, len(distances[i])):  # 最初のインデックスは自分自身なのでスキップ
                if distances[i][j] <= self.max_dist:
                    edges.append((valid_indices[i], valid_indices[indices[i][j]]))
        return torch.tensor(edges).t().contiguous()
        

    def forward(self, geo_ids):
        # グラフ畳み込み
        edge_index = self.edge_index.to(self.node_features.device)
        x = self.conv1(self.node_features, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        #print('x shape', x.shape)
        #print('geo ids max', geo_ids.max())
        return x[geo_ids.to(x.device)].bfloat16()#half()
    
    
        
    
    