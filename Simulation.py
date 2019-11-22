import numpy as np

def PlotGraph(edges,nodes):    
    indecesOfEdges = np.where(edges == 1)
    fromCityPositions = [nodes[indecesOfEdges[0],0],nodes[indecesOfEdges[0],1]]
    toCityPositions = [nodes[indecesOfEdges[1],0],nodes[indecesOfEdges[1],1]]
    
    plt.figure()
    plt.gca().set_aspect('equal', adjustable='box')
    markerSize = 10
    fontSize = 30
    lineWidth = 1
    plt.xlabel('x',fontsize=fontSize)
    plt.ylabel('y',fontsize=fontSize)
    plt.tick_params(axis='both', labelsize=fontSize)
    
    plt.plot([fromCityPositions[0][:],toCityPositions[0][:]],[fromCityPositions[1][:],toCityPositions[1][:]],'k',linewidth=lineWidth)
    plt.plot(nodes[:,0],nodes[:,1],'or',markersize=markerSize)