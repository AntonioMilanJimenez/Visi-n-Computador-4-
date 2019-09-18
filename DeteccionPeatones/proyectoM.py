# -*- coding: utf-8 -*-

import numpy as np
import cv2
import math
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import copy
import pickle
import os
import glob
import re
import collections as cts
from sklearn import svm
from sklearn.metrics import roc_curve, confusion_matrix, auc #area under curve
import random as rd
import gc

def draw_canvas(imagenes, etiquetas, f, c):

	item = 1
	tamanio = math.ceil(math.sqrt(len(imagenes)))

	for im, nom in zip(imagenes, etiquetas):
		plt.subplot(f, c, item)
		item = item + 1
		plt.title(nom)
		plt.imshow(im)#, cmap = "gray")

	plt.show()










#Función que, dado un directorio, el tamaño y el tipo de borde,
#carga las imágenes de dicho directorio y los etiquetas correspondientes.
def load_data(directory, vertical_border, horizontal_border, border_type):
	#Obtenemos los nombres de las imágenes contenidas en el
	#directorio dado como argumento
	pos = glob.glob(os.path.join(directory + "/pos", '*.png'))
	pos.extend(glob.glob(os.path.join(directory + "/pos", '*.jpg')))
	neg = glob.glob(os.path.join(directory + "/neg", '*.png'))
	neg.extend(glob.glob(os.path.join(directory + "/neg", '*.jpg')))
	#Cargamos las imágenes negativas
	neg_img = [cv2.imread(i) for i in neg]
	#Cargamos las imágenes positivas
	pos_img = [cv2.imread(i) for i in pos]
	#Submuestreamos aleatoriamente las imágenes negativas con una ventana de 64x128
	neg_img = get_negative_set(neg_img, (64, 128))
	#Submuestreamos aleatoriamente las imágenes positivas con una ventana de 64x128
	pos_img = get_positive_set(pos_img, (64, 128))
	#Añadimos un borde a las imágenes negativas en función
	#de los bordes especificados
	neg_img = [cv2.copyMakeBorder(i, vertical_border, vertical_border, \
				 horizontal_border, horizontal_border, border_type) for i in neg_img]
	#Añadimos un borde a las imágenes positivas en función
	#de los bordes especificados
	pos_img = [cv2.copyMakeBorder(i, vertical_border, vertical_border, \
				 horizontal_border, horizontal_border, border_type) for i in pos_img]
	#Creamos las etiquetas positivas en función del tamaño del conjunto positivo
	pos_labels = np.ones(len(pos_img), dtype = np.int)
	#Creamos las etiquetas positivas en función del tamaño del conjunto negativo
	neg_labels = np.zeros(len(neg_img), dtype = np.int)
	#Concatenamos las etiquetas positivas y negativas
	labels = np.append(pos_labels, neg_labels)
	#Concatenamos las imágenes positivas y negativas
	images = np.append(pos_img, neg_img, axis = 0)
	#Devolvemos las imágenes y las etiquetas
	return images, labels









#Función que, a partir del conjunto de imágenes negativas
#y un tamaño de ventana, realiza un submuestreo aleatorio
#de 10 imágenes del tamaño de la ventana para cada imagen
def get_negative_set(negative_images, win_size):
	neg_images = []
	#Recorremos las imágenes negativas, para las que
	#realizaremos el submuestro
	for ni in negative_images:
		for i in range(10):
			#Obtenemos aleatoriamente la fila
			rand1 = rd.randint(win_size[0]+1, np.shape(ni)[1])-win_size[0]-1
			#Obtenemos aleatoriamente la columna
			rand2 = rd.randint(win_size[1]+1, np.shape(ni)[0])-win_size[1]-1
			#Añadimos la muestra con la fila y columna obtenida y
			#del tamaño de ventana proporcionado
			neg_images.append(ni[rand2:rand2+win_size[1],\
								 rand1:rand1+win_size[0]])
	#Devolvemos todas las muestras calculadas de todas las
	#imágenes negativas
	return (np.asarray(neg_images)).reshape((len(neg_images), \
								win_size[1], win_size[0], 3))







#Función que, a partir del conjunto de imágenes positivas y un tamaño
#de ventana, extrae la subimagen central del tamaño de la ventana
def get_positive_set(positive_images, win_size):
	pos_images = []
	#Recorremos las imágenes positivas
	for pi in positive_images:
		#Calcula la fila central
		mid_rows = np.shape(pi)[0]//2
		#Calcula la columna central
		mid_cols = np.shape(pi)[1]//2
		#Añadimos la muestra central de la imagen del tamaño de
		#la ventana proporcionado
		pos_images.append(pi[mid_rows - win_size[1]//2:mid_rows + win_size[1]//2,\
						     mid_cols - win_size[0]//2:mid_cols + win_size[0]//2])
	#Devolvemos todas las muestras calculadas
	return np.array(pos_images)






#Función que centra y reescala una imagen dada como argumento
def center_scale(image):

	#Obtenemos el maximo
	maximo = image.max()
	#Obtenemos el minimo
	minimo = image.min()

	#Aplicamos el centrado y el reescalado
	return np.uint8((image - minimo.astype(np.float64))/(maximo.astype(np.float64) - minimo.astype(np.float64))*255)








#Función que aplica la corrección gamma a una lista de
#imagenes dada como argumento
def gamma_correction(img_list, gamma):
	#Elevamos cada imagen al parámetro gamma
	return [np.power(i, gamma, dtype = np.float32) for i in img_list]









#Función que obtiene los ángulos y las magnitudes de los gradientes
#de cada imagen de una lista dada como argumento
def get_angles_magnitudes(img_list):
	img_angles_list = []
	img_magnitudes_list = []
	#Iteramos sobre la lista de imágenes
	for i in img_list:
		#Obtenemos el kernel de Sobel
		kernel=np.array([-1,0,1])
		#Aplicamos el kernel a la imagen para obtener las derivadas x e y
		x_gradient = cv2.sepFilter2D(np.float32(i), cv2.CV_32F,kernel,np.array([1]))
		y_gradient = cv2.sepFilter2D(np.float32(i), cv2.CV_32F,np.array([1]),kernel)
		#Aplicamos la función cartToPolar para obtener las magnitudes
		#y orientación de cada pixel
		magnitudes, angles = cv2.cartToPolar(x_gradient, y_gradient,\
											 angleInDegrees=True)
		#De entre los tres valores de respuesta de un pixel, uno por cada canal,
		#obtenemos el máximo, así como el canal en el que fué encontrado
		maxs_index = np.argmax(magnitudes, axis=2)
		max_magnitudes = np.max(magnitudes, axis=2)
		#Inicializamos la lista de ángulos asociados a los mayores valores
		#de respuesta
		maxs_angles = np.zeros(np.shape(maxs_index), dtype = np.float32)
		#Iteramos sobre la imagen
		for i in range(maxs_index.shape[0]):
   			for j in range(maxs_index.shape[1]):
   				#Almacenamos el ángulo asociado al mayor valor de respuesta
   				maxs_angles[i][j] = angles[i][j][maxs_index[i][j]]
   		#Almacenamos la matriz de ángulos y magnitudes asociadas
		img_angles_list.append(maxs_angles)
		img_magnitudes_list.append(max_magnitudes)
	return img_angles_list, img_magnitudes_list












#Función que calcula el histograma de una matriz de ángulos y sus magnitudes
def get_histogram(angles_matrix, mag_matrix):

    #Inicializamos nuestro histograma de tamaño 9 a 0
    histogram = np.zeros(9, dtype = np.float32)

    #Recorremos la matriz de ángulos
    for i in range(angles_matrix.shape[0]):
        for j in range(angles_matrix.shape[1]):

            #Obtenemos los dos cubetas donde se repartirá el peso del ángulo
            bucket1 = 20 * (angles_matrix[i,j]//20)
            bucket2 = 20 * (angles_matrix[i,j]//20 + 1)

            #Calculamos la proporción de peso que recibirá la primera cubeta
            proportion = bucket2 - angles_matrix[i,j]

            #En función de la matriz de magnitudes y las propociones calculadas,
            #calculamos los pesos para cada cubeta respectivamente
            weight1 = (proportion/20)*mag_matrix[i,j]
            weight2 = ((20-proportion)/20)*mag_matrix[i,j]

            #Actualizamos las cubetas correspondientes con los pesos obtenidos
            histogram[int((bucket1/20)%9)] = histogram[int((bucket1/20)%9)] + weight1
            histogram[int((bucket2/20)%9)] = histogram[int((bucket2/20)%9)] + weight2

    return histogram









#Función que obtiene las características asociadas a las imágenes de las que se
#han obtenido sus matrices de ángulos y magnitudes, dados estos como argumento, y
#atendiento a los tamaños de celda y bloque dados
def get_features(angles_list, magnitudes_list, cell_size, block_size):
	#Controlamos que los parámetros con compatibles
	assert len(angles_list) == len(magnitudes_list)
	assert np.shape(angles_list[0])[0] % (cell_size) == 0 or \
		   np.shape(angles_list[0])[1] % (cell_size) == 0
	#Obtenemos la lista de ángulos en formato [0,180)
	ail = [i%180 for i in angles_list]
	aml = magnitudes_list
	#Calculamos el número de celdas por fila
	cpr = np.shape(ail[0])[0]//cell_size
	#Calculamos el número de celdas por columna
	cpc = np.shape(ail[0])[1]//cell_size
	#Inicializamos la lista de histogramas
	hist_list = []
	#Iteramos sobre la lista de matrices de ángulos
	for img in range(len(ail)):
		#Inicializamos el histograma asociado a la matriz
		hist_matrix = np.zeros((cpr,cpc,9), dtype = np.float32)
		#Iteramos sobre la matriz virtual de celdas
		for i in range(cpr):
			for j in range(cpc):
				#Obtenemos la sub-matriz de angulos asociada a la celda
				angles_cell = (ail[img])[i*cell_size:(i+1)*cell_size,\
										 j*cell_size:(j+1)*cell_size]
				#Obtenemos la sub-matriz de magnitudes asociada a la celda
				mag_cell = (aml[img])[i*cell_size:(i+1)*cell_size,\
									  j*cell_size:(j+1)*cell_size]
				#Obtenemos los histogramas asociados a cada celda almacenada
				hist_matrix[i,j] = get_histogram(angles_cell, mag_cell)
		#Almacenamos la matriz de histogramas calculada
		hist_list.append(hist_matrix)
	#Calculamos el margen asociado al bloque
	bm = block_size//2
	#Inicializamos la lista de características de imágenes
	img_features_list = []
	#Iteramos sobre la lista de matrices de histogramas
	for m_hist in hist_list:
		#Inicalizamos la matriz de vectores de características
		features_vector_matrix = []
		#Iteramos sobre la matriz de histogramas actual respetando los margenes
		for i in range(bm, m_hist.shape[0]- bm):
			for j in range(bm, m_hist.shape[1]- bm):
				#Obtenemos y almacenamos el vector de características
				#asociado a cada bloque, concatenando el vecindario
				#de histogramas que indique el tamaño de bloque.
				features_vector = (np.array(m_hist[i-bm:i+bm+1, j-bm:j+bm+1])).flatten()
				features_vector_matrix.append(features_vector)
		#Almacenamos la matriz de características de cada imagen
		img_features_list.append(features_vector_matrix)
	return img_features_list









#Normalización L2
def L2_norm(image_features_list, epsilon):
	#Iteramos sobre la lista de características de imágenes
	for features in image_features_list:
		#Iteramos sobre los bloques de la matriz de características
		for i in range(np.shape(features)[0]):
			#Aplicamos la normalización
			features[i] = features[i]/np.sqrt(np.linalg.norm(features[i])**2 + \
													epsilon**2)










#Normalización L1
def L1_norm(image_features_list, epsilon):
	#Iteramos sobre la lista de características de imágenes
	for features in image_features_list:
		#Iteramos sobre las filas de la matriz de características
		for i in range(np.shape(features)[0]):
			#Aplicamos la normalización
			features[i] = features[i]/(np.linalg.norm(features[i], 1) + \
													  epsilon)













#Normalización L1-sqrt
def L1_sqrt(image_features_list, epsilon):
	#Iteramos sobre la lista de características de imágenes
	for features in image_features_list:
		#Iteramos sobre las filas de la matriz de características
		for i in range(np.shape(features)[0]):
			#Aplicamos la normalización
			features[i] = np.sqrt(features[i]/(np.linalg.norm(features[i], 1) + \
															  epsilon))











#Función que transforma las características dadas como argumento al formato necesario
#para darlas como argumento a un SVM
def build_SVM_features(image_features_list):
	#Inicializamos la lista de características en formato SVM
	SVM_features = []
	#Iteramos sobre la lista de características dada como argumento
	for features in image_features_list:
		#Obtenemos la concatenación de las características contenidas en
		#las filas de la matriz de características asociada a cada imagen
		SVM_features.append(np.asarray(features).flatten())

	return np.array(SVM_features)







#Función que entrena una máquina de vectores de soporte
#dadas las características y etiquetas para ello
def train_SVM(features, labels):
	#Obtenemos el objeto SVM con kernel lineal
	SVM = svm.SVC(kernel='linear')
	#Ajustamos un modelo según las características y las
	#etiquetas dadas
	SVM.fit(features, labels)
	return SVM








#Función que obtiene falsos positivos asociados al SVM
#dado como argumento, prediciendo sobre las imágenes encontradas
#en el directorio dado como argumento
def hard_testing(dir,SVM,win_size):
	#Obtenemos la lista de nombres de las imágenes a analizar
	neg = glob.glob(os.path.join(dir, '*.png'))
	neg.extend(glob.glob(os.path.join(dir, '*.jpg')))
	#Inicializamos la lista de falsos positivos
	list_false_positive=[]
	#Iteramos sobre la lista de nombres
	for name_img in neg:
		#Cargamos la imagen
		neg_img = cv2.imread(name_img)
		#Aplicamos la corrección gamma a la imagen
		neg_img=gamma_correction([neg_img],0.2)[0]
		#Incialiamos la lista de ventanas a analizar
		list_windows=[]
		#Iteramos sobre la imagen respetanto los margenes asociados
		#a la imagen aplicando los saltos pertinentes
		for i in range(0, neg_img.shape[0] - win_size[1], 4):
			for j in range(0, neg_img.shape[1] - win_size[0], 8):
				#Extraemos la ventana, le añadimos los bordes y la almacenamos
				window = neg_img[i:i+win_size[1],j:j+win_size[0]]
				window = cv2.copyMakeBorder(window, 2, 2, 1, 1, cv2.BORDER_REPLICATE)
				list_windows.append(window)
		#Obtenemos los angulos y las magnitudes para cada imagen
		angles_list, mag_list = get_angles_magnitudes(list_windows)
		#Obtenemos las características no normalizadas
		unnormalized_features = get_features(angles_list, mag_list, 6, 3)
		#Normalizamos las características
		L2_norm(unnormalized_features, 0.95)
		#Obtenemos las características en formato compatible con SVM
		L2_norm_features = build_SVM_features(unnormalized_features)
		#Obtenemos las predicciones realizas con el modelos
		#sobre las ventanas procesadas
		L2_norm_predicted_labels = SVM.predict(L2_norm_features)
		#Obtenemos las ventanas correspondientes a los falsos
		#positivos mediante indexación por booleanos
		list_aux = np.asarray(list_windows)[np.asarray(L2_norm_predicted_labels, \
											dtype = np.bool)]
		#Almacenamos los falsos positivos
		list_false_positive.append(list_aux)
	return list_false_positive










#Función que detecta peatones presentes en una imagen dada como argumento,
#empleando como predictor el SVM dado como argumento
def recognise_pedestrian(image, win_size, SVM):
	#Aplicamos la corrección gamma a la imagen dada como argumento
    correc_image = gamma_correction([image],0.2)[0]
    #Obtenemos una copia sobre la que realizaremos modifcaciones
    output_image = copy.deepcopy(image)
    #Inicializamos la lista que contendrá la pirámide
    #gaussiana asociada a la imagen
    pyramid = [correc_image]
    r_size = pyramid[0].shape[0]
    c_size = pyramid[0].shape[1]
    #Mientras la imagen obtenida sea de dimensión mayor que la ventana
    while (r_size/1.2 > win_size[1] and c_size > win_size[0]):
    	#Almacenamos el siguiente nivel de la pirámide gaussiana
        pyramid.append(cv2.resize(cv2.GaussianBlur(pyramid[-1], (7,7), 1), \
        									(0, 0), fx = 0.75, fy = 0.75))
        #Actualizamos las variables centinela
        r_size = pyramid[-1].shape[0]
        c_size = pyramid[-1].shape[1]

    l = 0
    #Iteramos sobre los niveles de la pirámide
    for img in pyramid:
    	#Iteramos sobre cada imagen
        for i in range(0, img.shape[0] - win_size[1] + 1, 4):
            for j in range(0, img.shape[1] - win_size[0] + 1, 8):
            	#Extraemos la ventana a analizar
                window = img[i:i+win_size[1],j:j+win_size[0]]
                #Añadimos los bordes
                window = cv2.copyMakeBorder(window, 2, 2, 1, 1, cv2.BORDER_REPLICATE)
                #Obtenemos el kernel de Sobel
                kernel=np.array([-1,0,1])
                #Aplicamos el kernel a la imagen para obtener las derivadas x e y
                x_gradient = cv2.sepFilter2D(np.float32(window), \
                				cv2.CV_32F,kernel,np.array([1]))
                y_gradient = cv2.sepFilter2D(np.float32(window), \
                				cv2.CV_32F,np.array([1]),kernel)
                #Aplicamos la función cartToPolar para obtener las
                #magitudes y orientación de cada pixel
                magnitudes, angles = cv2.cartToPolar(x_gradient, y_gradient, \
                									 angleInDegrees=True)
                #De entre los tres valores de respuesta de un pixel, uno por cada canal,
					 #obtenemos el máximo, así como el canal en el que fué encontrado
                maxs_index = np.argmax(magnitudes, axis=2)
                max_magnitudes = np.max(magnitudes, axis=2)
                #Inicializamos la lista de angulos
                maxs_angles = np.zeros(np.shape(maxs_index), dtype = np.float32)
                #Iteramos sobre la imagen
                for a in range(maxs_index.shape[0]):
                    for b in range(maxs_index.shape[1]):
                    	#Almacenamos el ángulo asociado al mayor valor de respuesta
                        maxs_angles[a][b] = angles[a][b][maxs_index[a][b]]
                #Obtenemos las características asociadas a la imagen
                features = get_features([maxs_angles], [max_magnitudes], 6, 3)
                #Normalizamos las características
                L2_norm_features = copy.deepcopy(features)
                L2_norm(L2_norm_features, 0.95)
                #Obtenemos las características en formato compatible con SVM
                L2_norm_features = build_SVM_features(L2_norm_features)
                #Obtenemos la predicción del modelo
                result = SVM.predict(L2_norm_features)
                #Si la predicción es positiva dibujamos sobre la imagen
                #el marco asociado a la ventana actual
                if(result[0] == 1):
                    sc = 1.33**l
                    output_image = cv2.rectangle(output_image, (int(j*sc),int(i*sc)),\
                     (int(j*sc+win_size[0]*sc), int(i*sc + win_size[1]*sc)), \
                     (0,255,0), 2)
    	l = l + 1

    draw_canvas([output_image[:,:,::-1]], ["Pedestrians"], 1, 1)


def main():

	#Load train and test images and labels
    train_images, train_labels  = load_data("INRIAPerson/train_64x128_H96", 2, 1, cv2.BORDER_REPLICATE)
    test_images, test_labels  = load_data("INRIAPerson/test_64x128_H96", 2, 1, cv2.BORDER_REPLICATE)

	#Apply gamma correction to train and test images
    train_images = gamma_correction(train_images, 0.2)
    test_images = gamma_correction(test_images, 0.2)

	#Calculate angles and magnitudes of train and test images
    train_angles_list, train_mag_list = get_angles_magnitudes(train_images)
    test_angles_list, test_mag_list = get_angles_magnitudes(test_images)

    train_images=None
    test_images=None
    gc.collect()

	# Save train images angles list in pickle file
	# train_angles_list_file = open("saves/train_angles_list.pkl", "wb")
	# pickle.dump(train_angles_list, train_angles_list_file)
	# train_angles_list_file.close()

	# Save test images angles list in pickle file
	# test_angles_list_file = open("saves/test_angles_list.pkl", "wb")
	# pickle.dump(test_angles_list, test_angles_list_file)
	# test_angles_list_file.close()

	# Save train images magnitudes list in pickle file
	# train_mag_list_file = open("saves/train_mag_list.pkl", "wb")
	# pickle.dump(train_mag_list, train_mag_list_file)
	# train_mag_list_file.close()
	#
	# Save test images magnitudes list in pickle file
	# test_mag_list_file = open("saves/test_mag_list.pkl", "wb")
	# pickle.dump(test_mag_list, test_mag_list_file)
	# test_mag_list_file.close()

	#Calculate unormalized features for both test and train images sets
    train_unnormalized_features = get_features(train_angles_list, train_mag_list, 6, 3)
    test_unnormalized_features = get_features(test_angles_list, test_mag_list, 6, 3)

    train_angles_list=None
    train_mag_list=None
    test_angles_list=None
    test_mag_list=None
    gc.collect()

	# Save train unnormalized features in pickle file
	# train_unnormalized_features_file = open("saves/train_unnormalized_features.pkl", "wb")
	# pickle.dump(train_unnormalized_features, train_unnormalized_features_file)
	# train_unnormalized_features_file.close()
	#
	# Save test unnormalized features in pickle file
	# test_unnormalized_features_file = open("saves/test_unnormalized_features.pkl", "wb")
	# pickle.dump(test_unnormalized_features, test_unnormalized_features_file)
	# test_unnormalized_features_file.close()

	################ SVM trained with L2_norm normalization ################
    print("############ L2 Norm #############")

    L2_norm_features = copy.deepcopy(train_unnormalized_features)
    L2_norm(L2_norm_features, 0.95)
    L2_norm_features = build_SVM_features(L2_norm_features)

	# L2_norm_features_file = open("saves/L2_norm_features.pkl", "wb")
	# pickle.dump(L2_norm_features, L2_norm_features_file)
	# L2_norm_features_file.close()

    L2_norm_SVM = train_SVM(L2_norm_features, train_labels)

    L2_norm_SVM_file = open("saves/L2_norm_SVM_95_20.pkl", "wb")
    pickle.dump(L2_norm_SVM, L2_norm_SVM_file)
    L2_norm_SVM_file.close()

	################ SVM trained with L1_norm normalization ################
    print("############ L1 Norm #############")

    L1_norm_features = copy.deepcopy(train_unnormalized_features)
    L1_norm(L1_norm_features, 0.95)
    L1_norm_features = build_SVM_features(L1_norm_features)

	# L1_norm_features_file = open("saves/L1_norm_features.pkl", "wb")
	# pickle.dump(L1_norm_features, L1_norm_features_file)
	# L1_norm_features_file.close()

    L1_norm_SVM = train_SVM(L1_norm_features, train_labels)

    L1_norm_SVM_file = open("saves/L1_norm_SVM_95_20.pkl", "wb")
    pickle.dump(L1_norm_SVM, L1_norm_SVM_file)
    L1_norm_SVM_file.close()

	################ SVM trained with L1_sqrt normalization ################
    print("############ L1 Sqrt #############")

    L1_sqrt_features = copy.deepcopy(train_unnormalized_features)
    L1_sqrt(L1_sqrt_features, 0.95)
    L1_sqrt_features = build_SVM_features(L1_sqrt_features)

	# L1_sqrt_features_file = open("saves/L1_sqrt_features.pkl", "wb")
	# pickle.dump(L1_sqrt_features, L1_sqrt_features_file)
	# L1_sqrt_features_file.close()

    L1_sqrt_SVM = train_SVM(L1_sqrt_features, train_labels)

    L1_sqrt_SVM_file = open("saves/L1_sqrt_SVM_95_20.pkl", "wb")
    pickle.dump(L1_sqrt_SVM, L1_sqrt_SVM_file)
    L1_sqrt_SVM_file.close()

    train_unnormalized_features=None
    L2_norm_features=None
    L1_norm_features=None
    L1_sqrt_features=None
    gc.collect()

	#Once e have a trained SVM we can now measure its behave
	################ Measuring behave of L2_norm_SVM ################
    L2_norm_test_features = copy.deepcopy(test_unnormalized_features)
    L2_norm(L2_norm_test_features, 0.95)
    L2_norm_test_features = build_SVM_features(L2_norm_test_features)

    L2_norm_predicted_labels = L2_norm_SVM.predict(L2_norm_test_features)
    tn, fp, fn, tp = confusion_matrix(test_labels,L2_norm_predicted_labels).ravel()
    print((tn,fp,fn,tp))

	################ Measuring behave of L1_norm_SVM ################
    L1_norm_test_features = copy.deepcopy(test_unnormalized_features)
    L1_norm(L1_norm_test_features, 0.95)
    L1_norm_test_features = build_SVM_features(L1_norm_test_features)

    L1_norm_predicted_labels = L1_norm_SVM.predict(L1_norm_test_features)
    tn, fp, fn, tp = confusion_matrix(test_labels,L1_norm_predicted_labels).ravel()
    print((tn,fp,fn,tp))

	################ Measuring behave of L1_sqrt_SVM ################
    L1_sqrt_test_features = copy.deepcopy(test_unnormalized_features)
    L1_sqrt(L1_sqrt_test_features, 0.95)
    L1_sqrt_test_features = build_SVM_features(L1_sqrt_test_features)

    L1_sqrt_predicted_labels = L1_sqrt_SVM.predict(L1_sqrt_test_features)
    tn, fp, fn, tp = confusion_matrix(test_labels,L1_sqrt_predicted_labels).ravel()
    print((tn,fp,fn,tp))

	######### Plotting ROC curves #########
    fpr_l2_norm, tpr_l2_norm, threshold = roc_curve(test_labels, L2_norm_predicted_labels)
    fpr_l1_norm, tpr_l1_norm, threshold = roc_curve(test_labels, L1_norm_predicted_labels)
    fpr_l1_sqrt, tpr_l1_sqrt, threshold = roc_curve(test_labels, L1_sqrt_predicted_labels)

    L2_norm_auc = auc(fpr_l2_norm, tpr_l2_norm)
    L1_norm_auc = auc(fpr_l1_norm, tpr_l1_norm)
    L1_sqrt_auc = auc(fpr_l1_sqrt, tpr_l1_sqrt)

    plt.figure()
    plt.plot(fpr_l2_norm, tpr_l2_norm, color = "b", linestyle ='--', lw=3, label = 'ROC curve L2_norm (area = %0.2f)' % L2_norm_auc)
    plt.plot(fpr_l1_norm, tpr_l1_norm, color = "g", linestyle =':',  lw=3, label = 'ROC curve L1_norm (area = %0.2f)' % L1_norm_auc)
    plt.plot(fpr_l1_sqrt, tpr_l1_sqrt, color = "r", linestyle ='-',  lw=3, label = 'ROC curve L1_sqrt (area = %0.2f)' % L1_sqrt_auc)
    plt.legend(loc=4)
    plt.show()



if __name__ == "__main__": main()


# draw_canvas([(pos_original[0])[:,:,::-1], center_scale(pos[0])[:,:,::-1]], ["", ""], 1, 2)

    # SVM_file = open("saves/L2_norm_SVM_95.pkl", "rb")
    # L2_norm_SVM = pickle.load(SVM_file)
    # SVM_file.close()
	#
    # SVM_file = open("saves/L1_norm_SVM_95.pkl", "rb")
    # L1_norm_SVM = pickle.load(SVM_file)
    # SVM_file.close()
	#
    # SVM_file = open("saves/L1_sqrt_SVM_95.pkl", "rb")
    # L1_sqrt_SVM = pickle.load(SVM_file)
    # SVM_file.close()

	#image = cv2.imread("/home/antonio/Documentos/VC/Proyecto/INRIAPerson/Train/pos/crop001013.png")
#    image = cv2.imread("/home/antonio/Documentos/VC/Proyecto/INRIAPerson/Test/pos/person_280.png")
#    recognise_pedestrian(image,(64,128),L2_norm_SVM)

	# # list_false_true = hard_testing("INRIAPerson/train_64x128_H96/neg",L2_norm_SVM,(64,128))
