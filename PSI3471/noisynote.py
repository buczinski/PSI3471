import numpy as np
from sklearn import neighbors
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.preprocessing import StandardScaler

#leitura de arquivo
def le(nomearq):
    with open(nomearq,"r") as f:
       linhas=f.readlines()
    linha0=linhas[0].split()
    nl=int(linha0[0]); nc=int(linha0[1])
    a=np.empty((nl,nc),dtype=np.float32)
    for l in range(nl):
        linha=linhas[l+1].split()
        for c in range(nc):
            a[l,c]=np.float32(linha[c])
    return a


#leitura dos dados e normalização
np.random.seed(1)  # Para reprodutibilidade
ax=le("ax.txt"); ay=le("ay.txt")
qx=le("qx.txt"); qy=le("qy.txt")

scaler = StandardScaler()
axn = scaler.fit_transform(ax)
qxn = scaler.transform(qx)


#vizinho mais proximo
vizinho = neighbors.KNeighborsClassifier(n_neighbors=13, weights="uniform", algorithm="brute")
vizinho.fit(axn, ay.ravel())
qp = vizinho.predict(qxn)

erros = 0
for i in range(qp.shape[0]):
    if qp[i] != qy[i]: erros += 1
print("Vizinho mais proximo: Erros=%d/%d. Pct=%1.3f%%" % (erros, qp.shape[0], 100.0 * erros / qp.shape[0]))


#arvore de decisao
arvore= tree.DecisionTreeClassifier(criterion="entropy")
arvore= arvore.fit(axn, ay)
qp_t=arvore.predict(qxn)
erros_t=0
for j in range(qp_t.shape[0]):
    if qp_t[j]!=qy[j]: erros_t+=1
print("Arvore de decisão: Erros=%d/%d. Pct=%1.3f%%\n"%(erros_t,qp_t.shape[0],100.0*erros_t/qp_t.shape[0]))
