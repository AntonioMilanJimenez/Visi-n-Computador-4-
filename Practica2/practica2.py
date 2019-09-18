# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
from math import sqrt
from PIL import Image
from scipy import signal
import copy
    

#EJERCICIO 1

#A

#Función encargada de imprimir una lista de imagenes en color o escala de grises    
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


#Función encargada de reescalar los puntos en función de su escala original
def reescalarCord(puntos):

       
   
   for i in range(puntos.shape[0]):
       escala=puntos[i,2]
       puntos[i,0]=puntos[i,0]*escala
       puntos[i,1]=puntos[i,1]*escala  
     
   return puntos

#Función encargada de concatenar varias listas de puntos y ordenarlos en función de su valor Harris
def concatenarOrdenar(lista,tam):

    lista_puntos=np.concatenate((lista[0],lista[1]),axis=0)
    
    for i in range(tam-2):
        if(len(lista[i+2]) > 0):
            lista_puntos=np.concatenate((lista_puntos,lista[i+2]),axis=0)
    
        
    lista_puntos=lista_puntos[np.lexsort((lista_puntos[:,3],))]
    lista_puntos=lista_puntos[::-1]    
    lista_puntos=lista_puntos[0:500] 
    
    return lista_puntos
    

    
#Calcula el punto máximo en un entorno, comprobando si coincide con el indicado    
def detectar_maximo(imagen,centroY,centroX,blocksize):
    
    maximo=np.amax(imagen[centroY-blocksize:centroY+blocksize,centroX-blocksize:centroX+blocksize])  
    
    coincide=(maximo==imagen[centroY,centroX])
    
    return coincide
    


#Función encargada de poner a 0 todos los valores de una ventana en la imagen
def suprimir_valores(matriz,OrigventanaX,OrigventanaY,tamVX,tamVY):
    
    matriz[OrigventanaY-tamVY:OrigventanaY+tamVY,OrigventanaX-tamVX:OrigventanaX+tamVX]=0
    
    return matriz
    

#Función encargada de obtener los máximos de una imagen y eliminar los que no lo sean en su entorno    
def suprimir_no_maximos(imagen,blocksize,nivel_piramide,escala):
    
        lista_maximos=[]
        m_aux=np.ones((imagen.shape[0],imagen.shape[1]))        
        m_aux.fill(255)
        
        for i in range(blocksize,m_aux.shape[0]-blocksize):
            for j in range(blocksize,m_aux.shape[1]-blocksize):
                if(m_aux[i,j]==255):
                    if(detectar_maximo(imagen,i,j,int(blocksize/2))):
                          m_aux=suprimir_valores(m_aux,j,i,int(blocksize/2),int(blocksize/2))
                          lista_maximos.append([i,j,escala,imagen[i,j]])
         
        lista_maximos=np.array(lista_maximos) 
                 
        return lista_maximos
        

#Calcula aquellos puntos que su valor Harris es máximo en un entorno determinado
def calcular_Harris(img,blocksize,ksize,n_piramide,escala,reescalar=True,marco=True,criterio=0.04):
    
   mi_imagen_harris=cv2.cornerEigenValsAndVecs(img,blocksize,ksize)
   
   nueva_imagen=np.zeros((mi_imagen_harris.shape[0],mi_imagen_harris.shape[1]),dtype=np.float32)
   
   nueva_imagen=pow((mi_imagen_harris[:,:,0]*mi_imagen_harris[:,:,1])-criterio*(mi_imagen_harris[:,:,0]+mi_imagen_harris[:,:,1]),2)

   lista_maximos_coordenadas=suprimir_no_maximos(nueva_imagen,blocksize,n_piramide,escala)
   
       
   return nueva_imagen,lista_maximos_coordenadas 
 

#Función que contruye la pirámide Gaussiana de una imagen
def piramide_Gaussiana(img,n):
   
    lista_img_piramides = []
    img_mod = np.array(img, dtype=np.float32)
    

    for i in range(1,n+1):
        lista_img_piramides.append(img_mod)
        img_mod=cv2.pyrDown(img_mod)

    return lista_img_piramides
    
    



#######################
###Apartado B##########
#######################

#Función que refina la posición de una lista de puntos de una imagen
def refinarPosicion(img,puntos,blocksize):
    
    if(len(puntos)>0):        
        lista_tuplas=[]
        for punto in puntos:
                 tupla=(punto[0],punto[1])
                 lista_tuplas.append(tupla)
                 
                 
        nuevos_puntos=cv2.cornerSubPix(image=img,corners=np.array(lista_tuplas,dtype=np.float32),winSize=(int(blocksize/2),int(blocksize/2)),zeroZone=(-1,-1),criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,100,0.001))
        
        for i in range(len(puntos)):
            puntos[i,0]=nuevos_puntos[i,0]
            puntos[i,1]=nuevos_puntos[i,1]    

    return puntos
    

        


#######################
###Apartado C##########
#######################

#Función que pone todos los valores de una imagen en positivo y entre 0 y 255
def Reescalar(img,maximos=False,negativos=False):
    
    if(negativos):
        minimo=np.amin(img)
        img=img+minimo*-1
    
    maximo=np.amax(img)
    if(maximo>255 or maximos==True):
        coef=(float)(255/maximo)
        img=img*coef
     
       
    return img  

#Función que aplica un filtro Gaussiano mediante nucleos separables para imagenes en escala de grises
def Gauss_kernel_Grey(img,sigma,ind_kernel,kernel,borde):
    
    img_mod = np.array(img, dtype=np.float32)
    
    if(ind_kernel==0):
        if(sigma<1):
            tam=7
        else:    
            tam=int(sigma*6+1)
            
        kernel = cv2.getGaussianKernel(tam,sigma) 
    
    for i in range(len(img)):
        img_mod[i]=cv2.filter2D(img[i],cv2.CV_32F,np.flipud(kernel)).ravel()
     
 
    for i in range(len(img[0])):
        img_mod[:,i]=cv2.filter2D(img[:,i],cv2.CV_32F,np.flipud(kernel)).ravel()
        
      
    if(borde!=0):
        img_mod=cv2.copyMakeBorder(img_mod,10,10,10,10,borde)
        
    
    return img_mod        


#Función que realiza una convolucion con nucleos derivados, en x o y    
def Gauss_kernel_deriv(img,ind_deriv,kernel,kernel_no_derivado):
            
    #Necesario para que no elimine los valores negativos
    img_mod = np.array(img, dtype=np.float32)
       
    if(ind_deriv ==0):
        for i in range(len(img)):
            img_mod[i]=cv2.filter2D(img[i],cv2.CV_32F,np.flipud(kernel)).ravel()
        for i in range(len(img[0])):    
            img_mod[:,i]=cv2.filter2D(img_mod[:,i],cv2.CV_32F,np.flipud(kernel_no_derivado)).ravel()
            
    
    if(ind_deriv ==1):    
        for i in range(len(img_mod[0])):
            img_mod[:,i]=cv2.filter2D(img[:,i],cv2.CV_32F,np.flipud(kernel)).ravel()
        for i in range(len(img)):    
            img_mod[i]=cv2.filter2D(img_mod[i],cv2.CV_32F,np.flipud(kernel_no_derivado)).ravel()
    
      
    if(np.amin(img_mod)  < 0):              #Hay negativos
        img_mod = Reescalar(img_mod,False,True)
    
    return img_mod    
    

#Función que realiza la convolucion con nucleos derivados
def kernel_derivada(img,ordenX,ordenY,var,borde,sigma):
    
    img_mod = np.array(img, dtype=np.float32)    
    
    if(var==0):
        kernels=cv2.getDerivKernels(ordenX,0,int(sigma*6+1),normalize=True)
    else:
        kernels=cv2.getDerivKernels(0,ordenY,int(sigma*6+1),normalize=True)
 
    img_mod=Gauss_kernel_deriv(np.array(img, dtype=np.float32),var,kernels[var],kernels[(var+1)%2])
    
    if(sigma>1):
        
        n_kernel=cv2.getGaussianKernel(int(sqrt(sigma-1)*6+1),int(sqrt(sigma-1)))
        img_mod=Gauss_kernel_Grey(img_mod,int(sqrt(sigma-1)),1,n_kernel,0)
    
    if(borde!=0):
        img_mod=cv2.copyMakeBorder(img_mod,10,10,10,10,borde)


    return img_mod


#Función que calcula las orientaciones de todos los puntos de una imagen
def calcular_orientacion(img,sigma):
    
    imgX=kernel_derivada(img,1,0,0,0,sigma)
    imgY=kernel_derivada(img,0,1,1,0,sigma)

    angulos=np.arctan(imgY/imgX)
    angulos=np.degrees(angulos)

    return angulos
    

#Función que encuentra las orientaciones de una lista de puntos proporcionada    
def determinar_angulos(orientaciones,lista_puntos):
    
    l_angulos=[]
    for puntos in lista_puntos:
        n=math.log(puntos[2],2)
        orient_aux=orientaciones[int(n)]
        l_angulos.append(orient_aux[int(puntos[0]),int(puntos[1])])
        
    return l_angulos    
    
  
#A partir de una lista de puntos y sus angulos, crea la lista de keypoints  
def crear_kp(angulos,lista_puntos):
    
    l_kp=[]
    i=0
    
    for puntos in lista_puntos:
        kp=cv2.KeyPoint(puntos[1],puntos[0],puntos[2],angulos[i])
        l_kp.append(kp)
        i=i+1
        
        
    return np.array(l_kp)
    


#Función que calcula los "segundos" puntos, para poder posteriormente dibujar los segmentos
def crear_segmentos(angulos,lista_puntos):
    
    lista_segmentos=[]
    
    for i in range(lista_puntos.shape[0]):
        puntoX=lista_puntos[i,1]+lista_puntos[i,2]*5*np.cos(angulos[i])
        puntoY=lista_puntos[i,0]+lista_puntos[i,2]*5*np.sin(angulos[i])
        segmento=[(int(lista_puntos[i,1]),int(lista_puntos[i,0])),(int(puntoX),int(puntoY))]
        lista_segmentos.append(segmento)

    return lista_segmentos
    




#/////////////////////////
#EJERCICIO 2
#////////////////////////

#A partir de una imagen, calcula su lista de keypoints
def ObtenerKeyPoints(imagen,n_piramide,blocksize,ksize):
    
    
    mi_lista_imagenes=piramide_Gaussiana(imagen,n_piramide)
    
    
    lista_imagen=[]
    lista_puntos_separados=[]
    lista_imagen.append(imagen)
    
    for i in range(5):
        img=calcular_Harris(mi_lista_imagenes[i],blocksize,ksize,i+1,escala=pow(2,i))
        lista_puntos_separados.append(img[1])
        
   
    lista_puntos=concatenarOrdenar(lista_puntos_separados,n_piramide)
    lista_puntos=reescalarCord(lista_puntos)

    
    for i in range(len(lista_puntos_separados)):
        lista_puntos_separados[i]=refinarPosicion(mi_lista_imagenes[i],lista_puntos_separados[i],11)        
        
    lista_puntos=concatenarOrdenar(lista_puntos_separados,5) 
    lista_puntos=reescalarCord(lista_puntos)

    
    orientaciones=[]
    for i in range(len(mi_lista_imagenes)):
        orientaciones.append(calcular_orientacion(mi_lista_imagenes[i],sigma=5))
    
    lista_puntos=concatenarOrdenar(lista_puntos_separados,5)
    angulos=determinar_angulos(orientaciones,lista_puntos)
    lista_puntos=reescalarCord(lista_puntos)
    
    keypoints=crear_kp(angulos,lista_puntos)
    
    return keypoints
    

#Función que calcula los matches entre dos imágenes
def CalculaMatches(mi_imagen1,mi_imagen2,keypoints1,keypoints2,keypoints=False):
    
    sift=cv2.xfeatures2d.SIFT_create()
    
    if(keypoints==False):
        keypoints1, descriptores1=sift.detectAndCompute(mi_imagen1,None)
        keypoints2, descriptores2=sift.detectAndCompute(mi_imagen2,None)
    else:
        keypoints1,descriptores1=sift.compute(mi_imagen1,keypoints1)
        keypoints2,descriptores2=sift.compute(mi_imagen2,keypoints2)        
        
    
    matcherBF=cv2.BFMatcher(crossCheck=True)
    matches=matcherBF.match(descriptores1,descriptores2)
    matches = sorted(matches, key=lambda x:x.distance)
    img_bf = cv2.drawMatches(mi_imagen1,keypoints1,mi_imagen2,keypoints2,matches[0:40],None,flags=2)


    matcherKnn=cv2.BFMatcher()
    matchesKnn=matcherKnn.knnMatch(descriptores1,descriptores2,k=2)
            
    mejores=[]
    for a,b in matchesKnn:
        if (a.distance < 0.7*b.distance): #Lo escogemos sólo si es un 30% que el segundo mejor match.
            mejores.append(a)
            
    mejores = sorted(mejores, key=lambda x:x.distance)
    img_knn = cv2.drawMatches(mi_imagen1,keypoints1,mi_imagen2,keypoints2,mejores[0:40],None,flags=2) 
    
    if(len(matches)>len(mejores)):
        print("BF " + str(matches[len(mejores)-1].distance))
        print("Knn " + str(mejores[-1].distance))
    else:
        print("Knn " + str(mejores[len(matches)-1].distance))
        print("Bf " + str(matches[-1].distance))     
    
    return img_bf,img_knn
        


#///////////////////////////////////////////////
#Ejercicio 3 y 4
#//////////////////////////////////////////////


#Función que crea una imagen con las dimensiones de todas las dimensiones de todas las imágenes proporcionadas
def calcular_imgOut(lista_imagenes):
    
    intX=0
    intY=0
    
    for imagen in lista_imagenes:
        intX=intX+imagen.shape[1]
        intY=intY+imagen.shape[0]
        
    mi_img=np.zeros((intY,intX))
    mi_img.fill(255)
       
    return mi_img
 
   

#Calcula los matches entre dos imágenes y devuelve los puntos correspondientes    
def calcular_matches(imgA,imgB):

    sift=cv2.xfeatures2d.SIFT_create()
    keypoints1, descriptores1=sift.detectAndCompute(imgA,None)
    keypoints2, descriptores2=sift.detectAndCompute(imgB,None)

    
    matcherBF=cv2.BFMatcher(crossCheck=True)
    matches=matcherBF.match(descriptores1,descriptores2)
    
    matches = sorted(matches, key=lambda x:x.distance)     

    matches=matches[0:30]
    
    
    pts_A=[]
    pts_B=[]    

    for m in matches:
        pts_A.append(np.array(keypoints1[m.queryIdx].pt,dtype=np.float32))
        pts_B.append(np.array(keypoints2[m.trainIdx].pt,dtype=np.float32))
        
    pts_A=np.array(pts_A).reshape(-1,1,2)    
    pts_B=np.array(pts_B).reshape(-1,1,2)
    
   
    return pts_A,pts_B
    

#Función que calcula las homografías entre las imágenes y el mosaico con la imagen central    
def calcular_homografias(lista_imagenes,h_ini):
    
    tam=lista_imagenes.shape[0]
    homografias=[]

    for i in range(tam-1):
        ptsA,ptsB = calcular_matches(lista_imagenes[i],lista_imagenes[i+1])
        if(i<int(tam/2)):
            homografia, m=cv2.findHomography(ptsA,ptsB,cv2.RANSAC,1)
        else:
            homografia, m=cv2.findHomography(ptsB,ptsA,cv2.RANSAC,1)
        homografias.append(homografia)
        
    tamHom=len(homografias)    
        
        
    i=int(tamHom/2)
    while(i!=0):
        i=i-1
        if((i+1)==int(tamHom/2)):        
            homografias[i]=h_ini*homografias[i]
        else:
            homografias[i]=homografias[i+1]*homografias[i]
            
    
    i=int(tamHom/2)
    while(i!=(tamHom)):
        
        if(i==int(tamHom/2)):        
            homografias[i]=h_ini*homografias[i]
        else:
            homografias[i]=homografias[i-1]*homografias[i]        
        i=i+1        
       
    return homografias

#Función que elimina las filas y columnas de la imagen que esten completamente a 0
def eliminar_filas_columnas(img):
    
    filas_a_eliminar=[]
    columnas_a_eliminar=[]
    
    for i in range(img.shape[0]):    
        if (np.all(img[i] == [0,0,0])):
            filas_a_eliminar.append(i-len(filas_a_eliminar))            
                     
    for i in filas_a_eliminar:
        img=np.delete(img,i,axis=0)    
        
   
    for i in range(img.shape[1]):    
        if (np.all(img[:,i] == [0,0,0])):
            columnas_a_eliminar.append(i-len(columnas_a_eliminar))            
                     
    for i in columnas_a_eliminar:
        img=np.delete(img,i,axis=1)      
            
    return img        


#Función que crea el mosaico a partir de una lista de imágenes
def calcular_mosaico(lista_imagenes):
    
    img_out = calcular_imgOut(np.array(lista_imagenes))
    img_central=lista_imagenes[int((lista_imagenes.shape[0])/2)]
    
    centroXimg=int(img_out.shape[1]/2)
    centroYimg=int(img_out.shape[0]/2)    
    centroXimgCentral=int(img_central.shape[1]/2)
    centroYimgCentral=int(img_central.shape[0]/2)
    
    
    a=(centroXimg-centroXimgCentral,centroYimg-centroYimgCentral)
    
    h_ini=np.matrix([[1,0,a[0]],[0,1,a[1]],[0,0,1]],dtype=np.float32)
    img_out=cv2.warpPerspective(src=np.array(img_central),M=h_ini,dsize=(img_out.shape[1],img_out.shape[0]))
    
    homografias=calcular_homografias(lista_imagenes,h_ini)
    print(len(homografias))
        
    
    lista_imagenes=np.delete(lista_imagenes,int((lista_imagenes.shape[0])/2),axis=0)
    
 
    for i in range(lista_imagenes.shape[0]):      
        img_out=cv2.warpPerspective(src=np.array(lista_imagenes[i]),M=(homografias[i]),dst=img_out,dsize=(img_out.shape[1],img_out.shape[0]),borderMode=cv2.BORDER_TRANSPARENT)
        

    img_out=eliminar_filas_columnas(img_out)
    
    return img_out
    



def main():
    
    #----------------------------------------------------------------------------------------------------------------------
    # EJERCICIO 1
    #----------------------------------------------------------------------------------------------------------------------
    
    #Apartado A
    
    mi_imagen=cv2.imread("imagenes/Yosemite1.jpg",0)

    mi_lista_imagenes=piramide_Gaussiana(mi_imagen,5)
    
    
    lista_imagen=[]
    lista_puntos_separados=[]
    lista_imagen.append(mi_imagen)
    
    for i in range(5):
        img=calcular_Harris(mi_lista_imagenes[i],blocksize=11,ksize=7,n_piramide=i+1,escala=pow(2,i))
        lista_puntos_separados.append(img[1])
        
   
    lista_puntos=concatenarOrdenar(lista_puntos_separados,5)
    lista_puntos=reescalarCord(lista_puntos)
           
          
    titulos=["Ejercicio1 apartado A"]
    imprimir(copy.deepcopy(lista_imagen),1,1,titulos,lista_puntos,[],True)
    
    #Apartado B
    
    for i in range(len(lista_puntos_separados)):
        lista_puntos_separados[i]=refinarPosicion(mi_lista_imagenes[i],lista_puntos_separados[i],11)        
        
    lista_puntos=concatenarOrdenar(lista_puntos_separados,5) 
    lista_puntos=reescalarCord(lista_puntos)
   
    titulos=["Ejercicio1 apartado B"] 
    imprimir(copy.deepcopy(lista_imagen),1,1,titulos,lista_puntos,[],True)
    
    #Apartado C
    
    orientaciones=[]
    for i in range(len(mi_lista_imagenes)):
        orientaciones.append(calcular_orientacion(mi_lista_imagenes[i],sigma=5))
    
    lista_puntos=concatenarOrdenar(lista_puntos_separados,5)
    angulos=determinar_angulos(orientaciones,lista_puntos)
    lista_puntos=reescalarCord(lista_puntos)
    
    lista_keypoints=crear_kp(angulos,lista_puntos)
    segmentos=crear_segmentos(angulos,lista_puntos)
    
    titulos=["Ejercicio1 apartado C"]
    imprimir(copy.deepcopy(lista_imagen),1,1,titulos,lista_puntos,segmentos,True)    


    #Apartado D
    
    sift=cv2.xfeatures2d.SIFT_create()
    descriptores=sift.compute(mi_imagen,lista_keypoints)
    
    
    #----------------------------------------------------------------------------------------------------------------------
    # EJERCICIO 2
    #----------------------------------------------------------------------------------------------------------------------
    
    mi_imagen1=cv2.imread("imagenes/Yosemite1.jpg",0)
    mi_imagen2=cv2.imread("imagenes/Yosemite2.jpg",0)
    
    img_bf,img_knn=CalculaMatches(mi_imagen1,mi_imagen2,[],[])
    
    lista_imagenes_matches=[]
    lista_imagenes_matches.append(img_bf)
    lista_imagenes_matches.append(img_knn)
    
    imprimir(copy.deepcopy(lista_imagenes_matches),2,1,["BruteForce","Knn"],[],[],True)
    
     
    
    #----------------------------EXTRA------------------------------------------------------
    
    mi_imagen1=cv2.imread("imagenes/Yosemite1.jpg",0)
    mi_imagen2=cv2.imread("imagenes/Yosemite2.jpg",0)
    
    
    keypoints1 = ObtenerKeyPoints(mi_imagen1,5,11,7)
    keypoints2 = ObtenerKeyPoints(mi_imagen2,5,11,7)
    
    mi_img_bf,mi_img_knn=CalculaMatches(mi_imagen1,mi_imagen2,keypoints1,keypoints2,True)
    
    lista_imagenes_matches.append(mi_img_bf)
    lista_imagenes_matches.append(mi_img_knn)
    
    imprimir(copy.deepcopy(lista_imagenes_matches),2,2,["BruteForce","Knn","Mi_BruteForce","Mi_Knn"],[],[],True)

 
    #COMENTAR QUE AUNQUE EN LOS PEORES MATCHES SI QUE HAY DIFERENCIA, 
    #EN LOS MEJORES MATCHES QUE ESTAMOS UTILIZANDO, NO HAY DIFERENCIA EN USAR UNO U OTRO.
 
     
    #----------------------------------------------------------------------------------------------------------------------
    # EJERCICIO 3
    #----------------------------------------------------------------------------------------------------------------------
     
    lista_imgs=[]
        
    mi_img1=cv2.imread("imagenes/mosaico002.jpg")    
    mi_img2=cv2.imread("imagenes/mosaico003.jpg")
    mi_img3=cv2.imread("imagenes/mosaico004.jpg")
    
    lista_imgs.append(mi_img1)
    lista_imgs.append(mi_img2)
    lista_imgs.append(mi_img3)
    
    mosaico=calcular_mosaico(np.array(lista_imgs))    
    
    imprimir([copy.deepcopy(mosaico)],1,1,["Mosaico con 3 imagenes"],[],[])
    
    
    #----------------------------------------------------------------------------------------------------------------------
    # EJERCICIO 4
    #----------------------------------------------------------------------------------------------------------------------
    
    
    lista_imgs=[]
        
    mi_img1=cv2.imread("imagenes/mosaico002.jpg")    
    mi_img2=cv2.imread("imagenes/mosaico003.jpg")
    mi_img3=cv2.imread("imagenes/mosaico004.jpg")
    mi_img4=cv2.imread("imagenes/mosaico005.jpg")
    mi_img5=cv2.imread("imagenes/mosaico006.jpg")
    
    lista_imgs.append(mi_img1)
    lista_imgs.append(mi_img2)
    lista_imgs.append(mi_img3)
    lista_imgs.append(mi_img5)
    lista_imgs.append(mi_img4)  #Estan en diferente orden
    
    mosaico=calcular_mosaico(np.array(lista_imgs))    
    
    imprimir([copy.deepcopy(mosaico)],1,1,["Mosaico con 5 imagenes"],[],[])
    
    
    
    
if __name__ == "__main__": 
    main()          