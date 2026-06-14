import numpy as np
from src.convex_clustering import ConvexClusterer, knn_w
from src.convex_clustering.viz import animation_save

X = np.random.default_rng(0).normal(size=(6, 2))
W = knn_w(X, k=1)

model = ConvexClusterer(algorithm="ADMM", gamma=0.5)
model.fit(X, W)

print(model.labels_)
print(model.cluster_centers_.shape)

Uhist = model.centers_hist_
animation_save(X, Uhist,f =1,save_path="test.gif")