import sys
import numpy as np
from PIL import Image

# Get filenames for train and test set
train_names = set()
test_names = set()
for person_idx in range(1,41):
    for img_idx in range(1,8):
        train_names.add('{}_{}.png'.format(person_idx, img_idx))
    for img_idx in range(8,11):
        test_names.add('{}_{}.png'.format(person_idx, img_idx))

train_imgs = np.array([])
for filename in train_names:
#     print(filename)
    img = Image.open('{}/{}'.format(sys.argv[1], filename)).convert('L')
    img_as_np = np.asarray(img)
    if train_imgs.size == 0:
        train_imgs = img_as_np.reshape(-1,1)
    else:
        train_imgs = np.append(train_imgs, img_as_np.reshape(-1,1), axis=1)
#     print(img_as_np.shape)

train_average_face = np.average(train_imgs, axis=1).reshape(56,46)
train_cov_matrix = np.cov(train_imgs)
train_eig_vals, train_eig_vecs = np.linalg.eig(train_cov_matrix)
train_eig_vecs = train_eig_vecs.astype(float) # remove imaginary part

test_img = Image.open(sys.argv[2]).convert('L')
test_img_as_np = np.asarray(test_img).astype(float)

eigenfaces_used = train_imgs.shape[1] - 1
print('eigenfaces used:', eigenfaces_used)
# print('used', train_eig_vals[eigenfaces_used-1])
# print('not used', train_eig_vals[eigenfaces_used])
coeff = (test_img_as_np - train_average_face).reshape(-1).dot(train_eig_vecs[:,0:eigenfaces_used])
reconstructed_img = train_eig_vecs[:,0:eigenfaces_used].dot(coeff)
reconstructed_img = reconstructed_img.reshape(56,46) + train_average_face
# print(coeff.shape)
# print(reconstructed_img.shape)
print('reconstruction error:', np.mean(reconstructed_img - test_img_as_np)**2)
im = Image.fromarray((reconstructed_img).astype('uint8'))
im.save(sys.argv[3])

