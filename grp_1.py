from __future__ import division
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


# Creacion del grafo
G = nx.Graph()
# Nodos


# Uniones (edges)
ed = [(1, 2, 3), (1, 3, 3), (1, 4, 5),
      (2, 3, 2), (2, 5, 3), (3, 4, 2),
      (3, 5, 2), (4, 5, 5)]
G.add_weighted_edges_from(ed)


labels = nx.get_edge_attributes(G,'weight')
print(nx.info(G))
pos=nx.spring_layout(G)
nx.draw_networkx(G, pos=pos, with_labels=True)
#print nx.shortest_path(G, 1, 5, weight='weight')

nx.draw_networkx_edge_labels(G,pos=pos,edge_labels=labels)
#label = range(5)
#label1 = str(label)

#print label1



# Creacion Tau

Tau = np.zeros((5,5))
ed1 = np.array(ed)

for i,j,k in ed1:
    Tau[i - 1, j - 1] = k
    Tau[j - 1, i - 1] = k

#print Tau

# Creacion fermona
Fer = (Tau != 0) * 0.1

#print Fer

# Factor de evaporacion
rho = 0.2


# creacion de clase Hormiga
class Hormiga:
    path = []
    v = 1

    def __init__(self, id):
        self.id = id


nHormigas = 10

A = []
for i in range(nHormigas):
    A.append(Hormiga(i))

# DeltaTau
Dt = np.zeros_like(Fer)
iter = 200
TD = np.zeros(iter)
for inter in range(iter):
    dHormigas2 = np.zeros(nHormigas)
    for hor in range(10):

        # inicializacion para cada hormiga
        ini = 1
        fin = 5

        act = ini
        pre = 0
        path = []
        path1 = []
        cont = 0
        while act != fin:

            # Arreglo de hormigas


            nod = np.array(list(G.edges(nbunch=act, data='weight', default=1)))

            #print G.edges(2)
            #print nod
            pro = np.where(nod[:, 1] == pre)[0]
            nod = np.delete(nod, pro, 0)
            eta = 1/nod[:, 2]
            fr = Fer[nod[:, 0] - 1, nod[:, 1] - 1]
            eta = eta * fr
            #print eta
            mu = np.sum(eta)
            pr = np.cumsum(eta)/mu
            opc = nod[:, 1]
            x = np.random.random_sample()
            sl = np.where(pr > x)[0][0]
            ndu = np.array([nod[sl, 0], nod[sl, 1]])
            path.append(ndu)
            pre = act
            act = nod[sl, 1]
            dHormigas2[hor] += nod[sl, 2]
            #print act
            if cont >= 4:
                dHormigas2[hor] = 100
                break

        q = 0

        for i in range(len(path)):
            q = G[path[i][0]][path[i][1]]['weight'] + q

        #print q
        #print 1/q
        for i in range(len(path)):
            j = path[i][0] - 1
            k = path[i][1] - 1
            Dt[j, k] += 1/q
            Dt[k, j] += 1/q

    TD[inter] = np.sum(dHormigas2)
    #print Dt
    Fer = (1-rho) * Fer + Dt
    Dt = np.zeros_like(Fer)

print("Matriz de fermona resultante:")
print(Fer)
print("Camino a seguir:")
print(path)
plt.show()

plt.plot(TD)
plt.show()