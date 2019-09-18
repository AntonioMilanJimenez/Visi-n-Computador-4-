import cv2
import pickle
import numpy as np
from matplotlib import pyplot as plt
import math
from math import sqrt
from PIL import Image
from scipy import signal
import copy

#Función encargada de imprimir una lista de imágenes
def imprimir(imagenes,fila,col,titulos,lista_puntos,lista_segmentos,grises=False):
    
    n_img=1
    for img in imagenes:
    
        for puntos in lista_puntos:            
            cv2.circle(img,(int(puntos[1]),int(puntos[0])),int(puntos[2]*5),(0,255,0))
            
        for segmento in lista_segmentos:
            cv2.line(img,segmento[0],segmento[1],(0,0,0))
                     
        
        plt.subplot(fila,col,n_img)
        plt.subplots_adjust(hspace=0.8)
        if(grises==True):
            plt.imshow(img,cmap='gray')
        else:
            plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        plt.title(titulos[n_img-1])
        
                   
            
        n_img=n_img+1
    
    plt.show()


#Función encargada de cargar un diccionario a partir de un archivo    
def loadDictionary(filename):
    with open(filename,"rb") as fd:
        feat=pickle.load(fd)
    return feat["accuracy"],feat["labels"], feat["dictionary"]

#Función encargada de cargar descriptores y parches a partir de un archivo
def loadAux(filename, flagPatches):
    if flagPatches:
        with open(filename,"rb") as fd:
            feat=pickle.load(fd)
        return feat["descriptors"],feat["patches"]
    else:
        with open(filename,"rb") as fd:
            feat=pickle.load(fd)
        return feat["descriptors"]
    

#Función encargada de obtener los coordenadas de un recuadro de la imagen
def click_and_draw(event,x,y,flags,param):
    global refPt, imagen,FlagEND
    
    
   # if the left mouse button was clicked, record the starting
	# (x, y) coordinates and indicate that cropping is being
	# performed
    if  event == cv2.EVENT_LBUTTONDBLCLK:
        FlagEND= False
        cv2.destroyWindow("image")
        
    elif event == cv2.EVENT_LBUTTONDOWN:
        #refPt.append((x, y))
        #cropping = True
        print("rfePt[0]",refPt[0])
        FlagEND= False
        cv2.destroyWindow("image")

    elif (event == cv2.EVENT_MOUSEMOVE) & (len(refPt) > 0) & FlagEND:
    # check to see if the mouse move
        clone=imagen.copy()
        nPt=(x,y)
        print("npt",nPt)
        sz=len(refPt)
        cv2.line(clone,refPt[sz-1],nPt,(0, 255, 0), 2)
        cv2.imshow("image", clone)
        cv2.waitKey(0)
        
    elif event == cv2.EVENT_RBUTTONDOWN:
		# record the ending (x, y) coordinates and indicate that
		# the cropping operation is finished
        refPt.append((x, y))
        #cropping = False
        sz=len(refPt)
        print("refPt[sz]",sz,refPt[sz-1])
        cv2.line(imagen,refPt[sz-2],refPt[sz-1],(0, 255, 0), 2)
        cv2.imshow("image", imagen)
        cv2.waitKey(0)
        

#Función encargada de extraer una región de una imagen        
def extractRegion(image):
    global refPt, imagen,FlagEND
    imagen=image.copy()
    # load the image and setup the mouse callback function
    refPt=[]
    FlagEND=True
    #image = cv2.imread(filename)
    cv2.namedWindow("image")
    # keep looping until the 'q' key is pressed
    cv2.setMouseCallback("image", click_and_draw)
    #
    while FlagEND:
    	# display the image and wait for a keypress
        cv2.imshow("image", image)
        cv2.waitKey(0)
    #
    print('FlagEND', FlagEND)
    refPt.pop()
    refPt.append(refPt[0])
    cv2.destroyWindow("image")
    return refPt


# EJERCICIO 1-----------------------------------------------

    
#Función que calcula los matches entre dos imágenes    
def CalculaMatches(mi_imagen1,mi_imagen2, mascara,keypoints1,keypoints2):
    
    sift=cv2.xfeatures2d.SIFT_create()   
    
    keypoints1, descriptores1=sift.detectAndCompute(mi_imagen1,mascara)
    keypoints2, descriptores2=sift.detectAndCompute(mi_imagen2,None)
           

    matcherKnn=cv2.BFMatcher()
    matchesKnn=matcherKnn.knnMatch(descriptores1,descriptores2,k=2)
            
    mejores=[]
    for a,b in matchesKnn:
        if (a.distance < 0.7*b.distance): #Lo escogemos sólo si es un 30% que el segundo mejor match.
            mejores.append(a)
            
    mejores = sorted(mejores, key=lambda x:x.distance)
    img_knn = cv2.drawMatches(mi_imagen1,keypoints1,mi_imagen2,keypoints2,mejores[0:100],None,flags=2) 
         
    
    return img_knn
    


# EJERCICIO 2-----------------------------------------------

#Función que para cada clase tendremos sus descriptores correspondientes
def clasificarDescriptores(labels,descriptors):
    
    clasificados=[]
    for i in range(5000):
        clasificados.append([])
        
    for i in range(len(labels)):
        tupla=(descriptors[i],i)
        clasificados[labels[i,0]].append(tupla)
        
    return clasificados    

#Función que deEvuelve los patches más cercanos a cada uno de los centros    
def obtenerCercanos(descriptors,centers,patches,tamCenters):

    for i in range(len(descriptors)):
        for j in range(len(descriptors[i])):
            tupla=descriptors[i][j]
            dist=np.linalg.norm(tupla[0] - centers[i])
            descriptors[i][j]=descriptors[i][j]+(dist,) 
        descriptors[i]=sorted(descriptors[i],key=lambda tup: tup[2])
        descriptors[i]=descriptors[i][0:20]
   

    clasificadosPatches=[]
    for i in range(tamCenters):
        clasificadosPatches.append([])
        
    for i in range(len(clasificadosPatches)):
        for j in range(len(descriptors[i])):
           clasificadosPatches[i].append(patches[descriptors[i][j][1]])
           
    return descriptors,clasificadosPatches  

#Función encargada de calcular las varianzas entre los descriptores    
def obtenerVarianzas(descriptorsCercanos):
        
        for i in range(len(descriptorsCercanos)):
            for j in range(len(descriptorsCercanos[i])):
                descriptorsCercanos[i][j]=descriptorsCercanos[i][j][0]
    
        varianzas=np.zeros(len(descriptorsCercanos))
        for i in range(varianzas.shape[0]):
            varianzas[i]=np.mean(np.var(descriptorsCercanos[i],axis=0))
            
        return varianzas
       
#Función encargada de obtener los centroides que hayan tenido una mejor varianza        
def obtenerTop(varianzas,descriptorsCercanos,n):
    
    top=[]
    while len(top)<n:
        ind=np.argmin(varianzas)
        varianzas[ind]=100000000000000
        if(len(descriptorsCercanos[ind]) > 4):
            top.append(ind)    
  
    return top  
    
# EJERCICIO 3-----------------------------------------------    
    
#Función que a partir de una imagen, encuentra en un diccionario imágenes similares    
def buscar_similares(img,diccionario, indice_invertido, lista_histogramas): 
    
    #Construimos el histograma de nuestra imagen       
    lista_histogramas_unico=obtenerHistogramas([img],diccionario)
    histograma=lista_histogramas_unico[0]
    
    #Obtenemos aquellas listas de imagenes del indide invertido para las palabras que hemos encontrado en nuestra imagen
    imagenes_similares=[]    
    for i in range(histograma.shape[0]):
        if(histograma[i]>0):
            imagenes_similares.append(indice_invertido[i])
            
            
    #Construimos una especie de histograma contando el número de apariciones de cada imagen
    imagenes_finales=np.zeros(227)    
    for img_sim in imagenes_similares:            
        for i in range(len(img_sim)):
            img=img_sim[i]
            imagenes_finales[img]=imagenes_finales[img]+1


    #Construimos un "top100" con aquellas imágenes que más hayan aparecido, es decir, aquellas que más palabras tienen en común con nuestra imagen
    top100=[]
    imagenes_finales_aux=copy.deepcopy(imagenes_finales)
    for i in range(100):
        ind=np.argmax(imagenes_finales_aux)
        top100.append(ind)
        imagenes_finales_aux[ind]=-1


    #Obtenemos las simetrias entre cada uno de los histogramas del "top100" y el histograma de nuestra imagen
    simetrias=np.zeros(100)
    for i in range(100):
        simetrias[i]=np.dot(histograma,lista_histogramas[top100[i]])
        denom=np.linalg.norm(histograma)*np.linalg.norm(lista_histogramas[top100[i]])
        simetrias[i]=simetrias[i]/denom

    #Obtenemos las 6 imágenes con mejores simetrías
    top6=[]
    for i in range(6):
        ind=np.argmax(simetrias)
        top6.append(top100[ind])
        simetrias[ind]=-1

    return top6
    

#Función encargada de cargar en una lista las imágenes de un directorio
def cargarImagenes(tam):
       
    imagenes=[]
    
    for i in range(tam):  
        s="imagenes/" + str(i) + ".png"
        img=cv2.imread(s)
        imagenes.append(img)
   
    return imagenes
    
 
#Función encargada de obtener los histogramas de un conjunto de imágenes
def obtenerHistogramas(imagenes,diccionario):

    sift=cv2.xfeatures2d.SIFT_create(nfeatures=0,nOctaveLayers=4,contrastThreshold=0.0000000000000001)
    lista_histogramas=[]
    
    for mi_img in imagenes:
        keypoints, descriptores=sift.detectAndCompute(mi_img,None)
        nuevaM=np.dot(descriptores,np.transpose(diccionario))
        
        denominadores=np.apply_along_axis(np.linalg.norm,1,diccionario)
        for i in range(nuevaM.shape[1]):
            nuevaM[:,i]=nuevaM[:,i]/denominadores[i]
        
        histograma=np.zeros((5000))
        
        for descriptor in nuevaM:
            mejor_palabra=np.argmax(descriptor)
            histograma[mejor_palabra]=histograma[mejor_palabra]+1
        
        lista_histogramas.append(histograma)  
        
        
    return lista_histogramas    

#Función encargada de construir el indice invertido a partir de una lista de histogramas
def construirIndiceInvertido(lista_histogramas):
    

    indice_invertido=[]
    for i in range(5000):
        indice_invertido.append([])
        
    indice_img=0
    for histograma in lista_histogramas:
        for i in range(histograma.shape[0]):
            if(histograma[i]>0):
                indice_invertido[i].append(indice_img)
        indice_img=indice_img+1   

    return indice_invertido
 

    


def main():
    
    #Ejercicio 1
    
    mi_img1_1 = cv2.imread("imagenes/128.png")
    mi_img1_2 = cv2.imread("imagenes/130.png")
    
    mi_img2_1 = cv2.imread("imagenes/156.png")
    mi_img2_2 = cv2.imread("imagenes/157.png")
       
    mi_img3_1 = cv2.imread("imagenes/56.png")
    mi_img3_2 = cv2.imread("imagenes/60.png")
    
    points1=extractRegion(mi_img1_1)
    points2=extractRegion(mi_img2_1)
    points3=extractRegion(mi_img3_1)
    
    mask1=np.zeros((mi_img1_1.shape[0],mi_img1_1.shape[1],3))
    mask2=np.zeros((mi_img2_1.shape[0],mi_img2_1.shape[1],3))
    mask3=np.zeros((mi_img3_1.shape[0],mi_img3_1.shape[1],3))
    
    mask1=cv2.fillConvexPoly(mask1,np.array(points1,dtype=np.int32),(1.0,1.0,1.0))
    mask2=cv2.fillConvexPoly(mask2,np.array(points2,dtype=np.int32),(1.0,1.0,1.0))
    mask3=cv2.fillConvexPoly(mask3,np.array(points3,dtype=np.int32),(1.0,1.0,1.0))
    
    
    img_knn1=CalculaMatches(mi_img1_1,mi_img1_2,np.array(mask1[:,:,0],dtype=np.uint8),[],[])
    img_knn2=CalculaMatches(mi_img2_1,mi_img2_2,np.array(mask2[:,:,0],dtype=np.uint8),[],[])
    img_knn3=CalculaMatches(mi_img3_1,mi_img3_2,np.array(mask3[:,:,0],dtype=np.uint8),[],[])
    
    imprimir([img_knn1],1,1,["Futbolin 128-130"],[],[])
    imprimir([img_knn2],1,1,["Reloj 156-157"],[],[])
    imprimir([img_knn3],1,1,["Flores 56-60"],[],[])
     
    
    
    #Ejercicio 2
          
    
    descriptors,patches=loadAux("descriptorsAndpatches.pkl",1)

    compactness,labels,centers=loadDictionary("kmeanscenters5000.pkl")
    
    
    descriptorsClasificados=clasificarDescriptores(labels,descriptors)
    descriptorsCercanos,patchesCercanos=obtenerCercanos(descriptorsClasificados,centers,patches,len(centers))
    
    varianzas=obtenerVarianzas(descriptorsCercanos)
    
    top=obtenerTop(varianzas,descriptorsCercanos,2)            
                    

    lista_imgs1=[]
    lista_imgs2=[]
    titulos1=[]
    titulos2=[]    
    for i in range(2):
        for j in range(20):
            if(i==0):
                lista_imgs1.append(patchesCercanos[top[i]][j])
                titulos1.append("Centroide " + str(top[i]) + " Parche " + str(j))
            else:
                lista_imgs2.append(patchesCercanos[top[i]][j])
                titulos2.append("Centroide " + str(top[i]) + " Parche " + str(j))
            
    imprimir(lista_imgs1,5,4,titulos1,[],[])
    imprimir(lista_imgs2,5,4,titulos2,[],[])

        
        
    
    #Ejercicio 3
    
    diccionario=loadDictionary("kmeanscenters5000.pkl")
    diccionario=diccionario[2]
        
    imagenes=cargarImagenes(227)
    
    lista_histogramas=obtenerHistogramas(imagenes,diccionario) 
    
    indice_invertido=construirIndiceInvertido(lista_histogramas)          
        
    #Probamos con 3 imagenes-pregunta
    
    img1=cv2.imread("imagenes/95.png")
    img2=cv2.imread("imagenes/15.png")
    img3=cv2.imread("imagenes/200.png")
    top6_1=buscar_similares(img1,diccionario,indice_invertido, lista_histogramas)
    top6_2=buscar_similares(img2,diccionario,indice_invertido, lista_histogramas)
    top6_3=buscar_similares(img3,diccionario,indice_invertido, lista_histogramas)
    
    lista_imgs1=[]
    lista_imgs1.append(img1)
    titulos1=[]
    titulos1.append("Imagen Query " + str(top6_1[0]))   
    
    
    lista_imgs2=[]
    lista_imgs2.append(img2)
    titulos2=[]
    titulos2.append("Imagen Query " + str(top6_2[0]))
    
    
    lista_imgs3=[]
    lista_imgs3.append(img3)
    titulos3=[]
    titulos3.append("Imagen Query " + str(top6_3[0]))
    
    for i in range(3):
        for j in range(6):
            if(i==0):
                lista_imgs1.append(imagenes[top6_1[j]])
                titulos1.append("Imagen " + str(top6_1[j]))
            elif(i==1):
                lista_imgs2.append(imagenes[top6_2[j]])
                titulos2.append("Imagen " + str(top6_2[j]))
            else:
                lista_imgs3.append(imagenes[top6_3[j]])
                titulos3.append("Imagen " + str(top6_3[j]))
        
    imprimir(lista_imgs1,3,3,titulos1,[],[])
    imprimir(lista_imgs2,3,3,titulos2,[],[])
    imprimir(lista_imgs3,3,3,titulos3,[],[])

if __name__ == "__main__": 
    main() 




