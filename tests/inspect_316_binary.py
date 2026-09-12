import struct
import cv2
import numpy as np

path = 'Thermal image of equipment (Induction Motor) + 40 Ground Truths added/IR-Motor-bmp/A&B50/316.bmp'
with open(path, 'rb') as f:
    data = f.read()

print('=== ANALYSE BINAIRE DÉTAILLÉE DE 316.bmp ===')
print('Taille totale du fichier :', len(data), 'octets')

# 1. BITMAPFILEHEADER (14 octets)
magic, file_size, res1, res2, offset = struct.unpack('<2sIHHI', data[:14])
print('\n1. EN-TÊTE DE FICHIER (BITMAPFILEHEADER : octets 0 à 13)')
print('   - Signature (Magic)    : 0x{:02X} 0x{:02X} -> "{}" (Format Windows Bitmap)'.format(data[0], data[1], magic.decode('ascii')))
print('   - Taille fichier       : {} octets'.format(file_size))
print('   - Champs réservés      : {}, {}'.format(res1, res2))
print('   - Décalage pixels      : {} octets (les pixels commencent à l\'octet 54)'.format(offset))

# 2. BITMAPINFOHEADER (40 octets, de 14 à 53)
dib_size, w, h, planes, bpp, comp, img_size, x_res, y_res, n_colors, imp_colors = struct.unpack('<IIIHHIIIIII', data[14:54])
print('\n2. EN-TÊTE D\'INFORMATION (BITMAPINFOHEADER : octets 14 à 53)')
print('   - Taille du DIB header : {} octets (BITMAPINFOHEADER standard)'.format(dib_size))
print('   - Dimensions           : Largeur = {} px, Hauteur = {} px'.format(w, h))
print('   - Plans de couleur     : {}'.format(planes))
print('   - Profondeur (bpp)     : {} bits/pixel (8 bits Bleu, 8 bits Vert, 8 bits Rouge)'.format(bpp))
print('   - Compression          : {} (0 = BI_RGB, aucune compression)'.format(comp))
print('   - Taille brute image   : {} octets ({} lignes de {} octets)'.format(img_size, h, w * 3))
print('   - Résolution spatiale  : {} x {} px/mètre'.format(x_res, y_res))
print('   - Nombre de couleurs   : {} (0 = palette complète 16.7 millions de couleurs RVB)'.format(n_colors))

# 3. DONNÉES PIXELS (230 400 octets, de l'octet 54 à la fin)
pixel_data = data[offset:]
print('\n3. TABLEAU DES PIXELS (octets 54 à 230453)')
print('   - Nombre d\'octets      : {} (= 320 x 240 x 3)'.format(len(pixel_data)))
print('   - Ordre de stockage    : BGR, de bas en haut (Bottom-Up convention Windows BMP)')

# Échantillonnage de pixels
# En bas à gauche (fond ambiant froid)
sample_cold = list(pixel_data[:3])
# Au milieu de l'image (zone du moteur / point chaud)
img_bgr = cv2.imread(path)
h_img, w_img = img_bgr.shape[:2]
pixel_center = img_bgr[h_img // 2, w_img // 2] # (B, G, R)
pixel_max_loc = np.unravel_index(np.argmax(img_bgr.mean(axis=2)), (h_img, w_img))
pixel_hot = img_bgr[pixel_max_loc]

print('\n4. EXEMPLES DE PIXELS CONCRETS DANS 316.bmp :')
print('   - Pixel fond froid (coin)  : BGR = {} (très sombre / violet-noir)'.format(sample_cold))
print('   - Pixel centre carcasse    : BGR = {} (chaud / orange)'.format(list(pixel_center)))
print('   - Pixel point chaud max    : BGR = {} aux coordonnées y={}, x={}'.format(list(pixel_hot), pixel_max_loc[0], pixel_max_loc[1]))

print('\n5. VALEURS MAXIMALES ET MINIMALES DANS 316.bmp :')
print('   - Min par canal (B, G, R)  : ({}, {}, {})'.format(img_bgr[:,:,0].min(), img_bgr[:,:,1].min(), img_bgr[:,:,2].min()))
print('   - Max par canal (B, G, R)  : ({}, {}, {})'.format(img_bgr[:,:,0].max(), img_bgr[:,:,1].max(), img_bgr[:,:,2].max()))
