import numpy as np
import scipy.io as sio
import networkx as nx
import matplotlib.pyplot as plt

def sample(alpha):
    return np.random.dirichlet(alpha)


mtz = sio.loadmat('matriz inicial.mat')

mt2 = mtz['m2']

ps = mt2[:, :, 0]
#print(mt2[:, :, 1])
#print(mt2[:, :, 2])


class Hormiga:
    def __init__(self, id):
        self.id = id
        self.path = np.array([], dtype=np.uint8)
        self.distance = 10000

    def limpia(self):
        self.path = np.array([], dtype=np.uint8)

    def updatePath(self, x):
        if len(self.path) == 0:
            self.path = np.append(self.path, x)
        else:
            self.path = np.vstack((self.path, x))

    def updateDistance(self, d):
        self.distance = d

fil, col = np.shape(ps)
edges = []
for i in range(fil):
    for j in range(col):
        val = ps[i, j]
        if np.isfinite(val) and val > 1:
          edges.append([i+1,j+1,val])

l = len(edges)
edges = np.array(edges)
edges = edges.astype(int)
#print(np.array(np.resize(edges, (l,3))))


# Crear grafo y pasar los valores de distancias y conexiones
G = nx.Graph()
G.add_weighted_edges_from(edges)


# Plotear grafo
plt.figure(1)
labels = nx.get_edge_attributes(G,'weight')
#pos=nx.spring_layout(G)
pos=nx.circular_layout(G)
nx.draw_networkx(G, pos=pos, with_labels=True)
nx.draw_networkx_edge_labels(G,pos=pos,edge_labels=labels)


# Matriz
a = np.ones((25,25), dtype=np.uint16)



ini = 1
fin = 21

iter = 10
TD = np.zeros(iter)
nHormigas = 100


Hormigas = [Hormiga(i) for i in range(nHormigas)]

for inter in range(iter):
    # numero de hormigas
    dHormigas2 = np.zeros(nHormigas)
    for hor in range(nHormigas):

        # inicializacion para cada hormiga


        act = ini
        pre = 0
        path = []
        path1 = []
        # Contador de pasos
        cont = 0
        while act != fin:

            # Arreglo de hormigas


            nod = np.array(list(G.edges(nbunch=act, data='weight', default=1)))
            #nod = nod.astype(int)
            #print G.edges(2)
            #print nod
            pro = np.where(nod[:,1] == pre)[0]
            nod = np.delete(nod, pro, 0)
            
            fr = a[nod[:,0] -1, nod[:,1] -1]
            eta = sample(fr)
            #print eta
            selection = np.uint8(np.argmax(eta))
            sl = nod[selection, 1]
            dHormigas2[hor] += nod[selection, 2]

            ndu = np.array([act, sl])
            Hormigas[hor].updatePath(ndu)
            pre = act
            act = nod[selection, 1]
            cont +=  1

            #cont =+ 1
            if cont > 10:
                dHormigas2[hor] = 10000
                break
            #print act

    for hor in range(nHormigas):
        path2 = Hormigas[hor].path
        if dHormigas2[hor] < Hormigas[hor].distance:
            a1 = path2[:,0] -1
            a2 = path2[:,1] -1
            a[a1,a2] += 1
            Hormigas[hor].updateDistance(dHormigas2[hor])
        Hormigas[hor].limpia()

    #print path
    TD[inter] = np.sum(dHormigas2)

#print "Matriz de fermona resultante:"
#print Fer


#Obtener camino resultante


print("Camino a seguir:")
print(path2)


plt.figure(2)
plt.plot(TD/nHormigas)
plt.show()
