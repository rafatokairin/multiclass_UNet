import cv2
import numpy as np
from keras.utils import normalize
from matplotlib import pyplot as plt
from simple_multi_unet import multi_unet # Usa softmax
from patchify import patchify, unpatchify

# Definir o tamanho das imagens e quantidade de classes da segmentação
SIZE_X = 128
SIZE_Y = 128
n_classes = 4

# Definir o limiar adequado com base na confiança da segmentação
segmentation_threshold = 2

# Função para criar a máscara binária usando um limiar
def create_binary_mask(segmented_image, threshold):
    return (segmented_image > threshold).astype(np.uint8) * 255

# Função para carregar pesos de epochs_50.hdf5
def load_trained_model(weights_path):
    model = multi_unet(n_classes = n_classes, IMG_HEIGHT = SIZE_X, IMG_WIDTH = SIZE_Y, IMG_CHANNELS = 1)
    model.load_weights(weights_path)
    return model

# Carregar o modelo treinado
model = load_trained_model('epochs_50.hdf5')

# Fazer a segmentação da imagem
img = cv2.imread('images/img.tif', 0)
patch = img[:, :, None]
patch_norm = np.expand_dims(patch, 0)
img_input = normalize(patch_norm, axis = 1)
prediction = model.predict(img_input)
predicted_img = np.argmax(prediction, axis = 3)[0, :, :]

# Criar a máscara binária
binary_mask = create_binary_mask(predicted_img, segmentation_threshold)

plt.figure(figsize = (18, 6))
plt.subplot(331)
plt.title('Imagem Original')
plt.imshow(img, cmap = 'gray')
plt.subplot(332)
plt.title('Imagem Segmentada U-Net')
plt.imshow(predicted_img, cmap = 'jet') # Usando o mapa de cores "jet"
plt.colorbar()
plt.subplot(333)
plt.title('Máscara Binária U-Net')
plt.imshow(binary_mask, cmap = 'gray')
plt.tight_layout()
plt.show()

# Dividir imagem grande em subimagens 128x128px, segmentá-las e concatená-las.
large_image = cv2.imread('images/large_img.tif', 0)
patches = patchify(large_image, (128, 128), step = 128)
predicted_patches = []
for i in range(patches.shape[0]):
    for j in range(patches.shape[1]):
        print(i,j)
        single_patch = patches[i, j, :, :]       
        single_patch_norm = np.expand_dims(normalize(np.array(single_patch), axis = 1), 2)
        single_patch_input=np.expand_dims(single_patch_norm, 0)
        single_patch_prediction = (model.predict(single_patch_input))
        single_patch_predicted_img=np.argmax(single_patch_prediction, axis = 3)[0, :, :]
        predicted_patches.append(single_patch_predicted_img)
predicted_patches = np.array(predicted_patches)
predicted_patches_reshaped = np.reshape(predicted_patches, (patches.shape[0], patches.shape[1], 128, 128) )
reconstructed_image = unpatchify(predicted_patches_reshaped, large_image.shape)

# Criar a máscara binária
binary_mask_large = create_binary_mask(reconstructed_image, segmentation_threshold)

plt.figure(figsize = (8, 8))
plt.subplot(331)
plt.title('Imagem Original')
plt.imshow(large_image, cmap = 'gray')
plt.subplot(332)
plt.title('Imagem Segmentada U-Net')
plt.imshow(reconstructed_image, cmap = 'jet')
plt.colorbar()
plt.subplot(333)
plt.title('Máscara Binária U-Net')
plt.imshow(binary_mask_large, cmap = 'gray')
plt.tight_layout()
plt.show()
