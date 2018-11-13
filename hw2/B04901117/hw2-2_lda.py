import sys
from PIL import Image
import numpy as np
from numpy.linalg import inv

train_imgs = np.zeros((40,7,2576))
for c in range(40):
    for n in range(7):
        img = Image.open('{}/{}_{}.png'.format(sys.argv[1],c+1,n+1)).convert('L')
        img_as_np = np.asarray(img)
        train_imgs[c][n] = img_as_np.reshape(-1)

train_average_face = train_imgs.mean(axis=(0,1))
train_cov_matrix = np.cov(train_imgs.reshape(-1,2576).T)
train_eig_vals, train_eig_vecs = np.linalg.eig(train_cov_matrix)
train_eig_vecs = train_eig_vecs.astype(float)

C = 40
N = 280
eigenfaces_used = N-C
train_img_PCA_coeff = (train_imgs - train_average_face).dot(train_eig_vecs[:,0:eigenfaces_used])

S_W = np.zeros((eigenfaces_used,eigenfaces_used))
for c in range(len(train_imgs)):
    class_mean = train_img_PCA_coeff[c].mean(axis=0)
    x_minus_u = train_img_PCA_coeff[c] - class_mean
    S_W += (x_minus_u.T).dot(x_minus_u)

PCA_global_mean = train_img_PCA_coeff.mean(axis=(0,1))
u_class_minus_u_global = train_img_PCA_coeff.mean(1) - PCA_global_mean
S_B = (u_class_minus_u_global.T).dot(u_class_minus_u_global)

LDA_eig_vals, W = np.linalg.eig(inv(S_W).dot(S_B))
W = W.astype(float)

fisherfaces = train_eig_vecs[:,0:eigenfaces_used].dot(W[:,0:C-1])

for i in range(1):
    pixel_range = fisherfaces[:,i].max() - fisherfaces[:,i].min()
    pixel_delta = fisherfaces[:,i] - fisherfaces[:,i].min()
    pixel_normalized = pixel_delta * 255 / pixel_range
    im = Image.fromarray((pixel_normalized).astype('uint8').reshape(56,46))
    im.save(sys.argv[2])