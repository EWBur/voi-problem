from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import osmnx as ox
import networkx as nx
import numpy as np

G = ox.core.graph_from_bbox(
    57.6943, 57.6852, 11.9832, 11.9721, network_type='drive', simplify=True)

ox.plot_graph(G)
fig, ax = plt.subplots()
node_list = G.nodes()
edge_nodes = G.edges()
lines = []
A = nx.adjacency_matrix(G).todense()
node_Xs = [float(x) for _, x in G.nodes(data='x')]
node_Ys = [float(y) for _, y in G.nodes(data='y')]
ax.scatter(node_Xs,node_Ys,s=10)
adjMat = np.zeros(shape=A.shape) ##Weighted Adjacency Matrix
for u, v in edge_nodes:
    data = min(G.get_edge_data(u, v).values(), key=lambda x: x['length'])
    if 'geometry' in data:
        xs, ys = data['geometry'].xy
        l = data['geometry'].length
        print(l)
        lines.append(list(zip(xs, ys)))
    else:
        x1 = G.nodes[u]['x']
        y1 = G.nodes[u]['y']
        x2 = G.nodes[v]['x']
        y2 = G.nodes[v]['y']
        line = [(x1, y1), (x2, y2)]
        l = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        lines.append(line)
    nodes = [n for n in G.nodes()]
    uIndex = nodes.index(u)
    vIndex = nodes.index(v)
    adjMat[uIndex,vIndex] = l


lc = LineCollection(lines)
ax.add_collection(lc)

#nx.draw(G)
plt.show()

