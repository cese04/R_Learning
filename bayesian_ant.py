import matplotlib.pyplot as plt
import networkx as nx
import numpy as np



n_nodos = 5

class Bandit2:
    def __init__(self, p):
        self.p = p
        self.a = 1
        self.b = 1

    def pull(self):
        return np.random.random() < self.p

    def sample(self):
        return np.random.beta(self.a, self.b)

    def update(self, x):
        self.a += 1 - x
        self.b += x


def sample(a, b):
    return np.random.beta(a, b)


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
pos = nx.spring_layout(G)
nx.draw_networkx(G, pos=pos, with_labels=True)
#print nx.shortest_path(G, 1, 5, weight='weight')

nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=labels)


# Creacion Tau

Tau = np.zeros((n_nodos, n_nodos))
ed1 = np.array(ed)

for i, j, k in ed1:
    Tau[i - 1, j - 1] = k
    Tau[j - 1, i - 1] = k

# Parametros de distribucion beta

a = np.ones((n_nodos, n_nodos))
b = np.ones((n_nodos, n_nodos))

# Factor de evaporacion
rho = 0.05


class Hormiga:
    def __init__(self, id):
        self.id = id
        self.path = np.array([], dtype=np.uint8)
        self.distance = 1000

    def limpia(self):
        self.path = np.array([], dtype=np.uint8)

    def updatePath(self, x):
        if len(self.path) == 0:
            self.path = np.append(self.path, x)
        else:
            self.path = np.vstack((self.path, x))

    def updateDistance(self, d):
        self.distance = d


nHormigas = 10
Hormigas = [Hormiga(i) for i in range(nHormigas)]


# DeltaTau
Dt = np.zeros_like(Tau)
iter = 200
TD = np.zeros(iter)
for inter in range(iter):
    dHormigas2 = np.zeros(nHormigas)
    for hor in range(nHormigas):

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
            #print(nod)
            #print G.edges(2)
            #print nod
            pro = np.where(nod[:, 1] == pre)[0]
            nod = np.delete(nod, pro, 0)

            eta = sample(np.uint8(a[act - 1, nod[:, 1] - 1]), np.uint8(b[act - 1, nod[:, 1] - 1]))
            selection = np.uint8(np.argmax(eta))
            sl = nod[selection, 1]
            dHormigas2[hor] += nod[selection, 2]

            ndu = np.array([act, sl])
            Hormigas[hor].updatePath(ndu)
            pre = act
            act = nod[selection, 1]
            cont = + 1
            #print(Hormigas[hor].path)

            if cont >= 4:
                dHormigas2[hor] = 100
                break
            #print act
        path2 = Hormigas[hor].path
        print(hor, path2)
        #print(hor, path2[0,:])
    #print(path2[:, 0])
    TD[inter] = np.sum(dHormigas2)

    for hor in range(nHormigas):
        path2 = Hormigas[hor].path
        
        if dHormigas2[hor] <= Hormigas[hor].distance:
            #print('Mejora')
            a1 = np.array(path2[0, :] - 1, dtype=np.uint8)
            a2 = np.array(path2[1, :] - 1, dtype=np.uint8)
            a[a1, a2] += 1
            a = np.ceil(0.99 * a)
            Hormigas[hor].updateDistance(dHormigas2[hor])
            Hormigas[hor].limpia()
            
        else:
            #print('Peor')
            a1 = np.array(path2[0, :] - 1, dtype=np.uint8)
            a2 = np.array(path2[1, :] - 1, dtype=np.uint8)
            b[a1, a2] += 1
            b = np.ceil(0.99 * b)
            Hormigas[hor].limpia()


        #q = 0

        #for i in range(len(path)):
        #    q = G[path[i][0]][path[i][1]]['weight'] + q

        #print q
        #print 1/q
        #for i in range(len(path)):
        #    j = path[i][0] - 1
        #    k = path[i][1] - 1
        #    Dt[j, k] += 1/q
        #    Dt[k, j] += 1/q


    #print Dt
    #Fer = (1-rho) * Fer + Dt
    #Dt = np.zeros_like(Fer)

print("Matriz de fermona resultante:")
print(a)
print(b)
print("Camino a seguir:")
print(Hormigas[0].path)
plt.show()

plt.plot(TD)
plt.show()