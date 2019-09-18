# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
from math import sqrt,pi,exp,pow
from PIL import Image
from scipy import signal
from copy import deepcopy
    

#EJERCICIO 1

#A

#Función encargada de imprimir una lista de imagenes en color o escala de grises    
def imprimir(imagenes,fila,col,titulos,grises=False):
    
    n_img=1
    for img in imagenes:
        plt.subplot(fila,col,n_img)
        plt.subplots_adjust(hspace=0.8)
        if(grises==True):
            plt.imshow(img,cmap='gray')
        else:
            plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        plt.title(titulos[n_img-1])    
        n_img=n_img+1
    
    plt.show()

 

#B

#Función que aplica una máscara Gaussiana sobre una imagen
def Gauss_alisamiento(img,sigmaX,sigmaY,borde):
    
    if(sigmaX<1):
        tamX=7
    else:    
        tamX=int(sigmaX*6+1)
        
    if(sigmaY<1):
        tamY=7
    else:    
        tamY=int(sigmaY*6+1)   
        
    img_mod=np.zeros((img.shape[0],img.shape[1],3),dtype=np.float32)    
        
    img_mod=cv2.GaussianBlur(np.array(img, dtype=np.float32),(tamX,tamY),sigmaX=sigmaX,sigmaY=sigmaY)
    img_mod=cv2.copyMakeBorder(img_mod,10,10,10,10,borde)
    
    return img_mod
    

#C


#Ind_kernel:
#0 --> No hay kernel
#1 --> Proporcionamos kernel


#Función que aplica un filtro Gaussiano mediante nucleos separables
def Gauss_kernel_RGB(img,sigma,ind_kernel,kernel,borde):
    
    img_mod = np.array(img, dtype=np.float32)
        
    if(ind_kernel==0):
        if(sigma<1):
            tam=7
        else:    
            tam=int(sigma*6+1)
            
        kernel = cv2.getGaussianKernel(tam,sigma)    
           
    
    for i in range(len(img)):
        img_mod[i]=cv2.filter2D(img[i],cv2.CV_32F,np.flipud(kernel))
        

    for i in range(len(img[0])):
        img_mod[:,i]=cv2.filter2D(img[:,i],cv2.CV_32F,np.flipud(kernel))
        
    if(borde!=0):
        img_mod=cv2.copyMakeBorder(img_mod,10,10,10,10,borde) 
           
    return img_mod
    
    
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

      

#--------------------------------------------------------------------------
#D
#--------------------------------------------------------------------------


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
    
    
#ind_deriv->0 respecto x
#ind_deriv->1 respecto y   


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
        kernels=cv2.getDerivKernels(ordenX,0,sigma*6+1,normalize=True)
    else:
        kernels=cv2.getDerivKernels(0,ordenY,sigma*6+1,normalize=True)
 
    img_mod=Gauss_kernel_deriv(np.array(img, dtype=np.float32),var,kernels[var],kernels[(var+1)%2])
    
    if(sigma>1):
        
        n_kernel=cv2.getGaussianKernel(int(sqrt(sigma-1)*6+1),int(sqrt(sigma-1)))
        img_mod=Gauss_kernel_Grey(img_mod,int(sqrt(sigma-1)),1,n_kernel,0)
    
    if(borde!=0):
        img_mod=cv2.copyMakeBorder(img_mod,10,10,10,10,borde)


    return img_mod
    

#--------------------------------------------------------------------------
#F
#--------------------------------------------------------------------------


#Función que realiza una convolucion con nucleo de Laplaciana-de-Gaussiana
def Laplaciana_Gaussiana(img,borde,sigma):
    
    img_X = np.array(img, dtype=np.float32)    
    img_Y = np.array(img, dtype=np.float32)

    img_X=kernel_derivada(np.array(img, dtype=np.float32),2,0,0,0,sigma)      
    img_Y=kernel_derivada(np.array(img, dtype=np.float32),0,2,1,0,sigma)     
    
    
    img_mod=img_X + img_Y
    img_mod=img_mod*pow(sigma,2) #Normalizamos
    
    if(np.amin(img_mod)<0):
        img_mod=Reescalar(img_mod,False,True)
    else:
        img_mod=Reescalar(img_mod)
    
    if(borde!=0):
        img_mod=cv2.copyMakeBorder(img_mod,10,10,10,10,borde)
        
        
    return img_mod     


#--------------------------------------------------------------------------
#G
#--------------------------------------------------------------------------


#Función que crea una imagen a la que va añadiendo una lista de imagenes para crear una estructura de pirámide
def construir_piramide(lista_imagenes,colores=False):
    
    tamY=(lista_imagenes[0].shape)[0]
    tamX=int((lista_imagenes[0].shape)[1]*2+1)
    
    if(colores==True):
        piramide=np.ones((tamY,tamX,3),dtype=np.float32)
    else:
        piramide=np.ones((tamY,tamX),dtype=np.float32)
        
    
    limiteX=0
    
    for i in range(len(lista_imagenes)):    
        x= (lista_imagenes[i].shape)[1]
        y= (lista_imagenes[i].shape)[0]
        
        piramide[0:y,limiteX:x+limiteX]=lista_imagenes[i]
        
        limiteX=x+limiteX

    return piramide
    
    

#Función que crea una lista de imagenes reducidas a partir de una original    
def piramide_Gaussiana(img,borde,n):

    
    lista_img_piramides = []
    img_mod = np.array(img, dtype=np.float32)
    img_aux=np.array(img_mod, dtype=np.float32)

    for i in range(1,n+1):
        img_aux=cv2.copyMakeBorder(img_mod,int(16/pow(2,i-1)),int(16/pow(2,i-1)),int(16/pow(2,i-1)),int(16/pow(2,i-1)),borde)
        lista_img_piramides.append(img_aux)
        img_mod=cv2.pyrDown(img_mod)

    return lista_img_piramides
    


#--------------------------------------------------------------------------
#H
#--------------------------------------------------------------------------


#Función que crea una piramide Laplaciana
def piramide_Laplaciana(img,borde,n):
   
    
    lista_img_piramides = []
    img_mod = np.array(img, dtype=np.float32)
    img_r = np.array(img, dtype=np.float32)
    img_a = np.array(img, dtype=np.float32)
    img_lap = np.array(img, dtype=np.float32)
    img_aux=np.array(img_mod, dtype=np.float32)

    for i in range(1,n+1):   
        
        if(img_mod.shape[0]%2==1):
            img_mod=img_mod[:-1,:]
        
        if(img_mod.shape[1]%2==1):
            img_mod=img_mod[:,:-1]        
        
        img_r=cv2.pyrDown(img_mod)
        img_a=cv2.pyrUp(img_r)
        img_lap=img_mod-img_a
        img_lap=Reescalar(img_lap,False,True)
        img_aux=cv2.copyMakeBorder(img_lap,int(16/pow(2,i-1)),int(16/pow(2,i-1)),int(16/pow(2,i-1)),int(16/pow(2,i-1)),borde)
        lista_img_piramides.append(img_aux)
        img_mod=img_r

    return lista_img_piramides
    


#------------------------------------------------------------------
# Apartado B: Imágenes Híbridas
#------------------------------------------------------------------

    
#Función que se encarga de todo el proceso para obtener las imágenes híbridas

def Get_Imagenes_Hibridas(img_alta,img_baja,sigma_alto,sigma_bajo):
    
    mi_lista=[]
    img_paso_alto = np.array(img_alta, dtype=np.float32)
    img_paso_bajo = np.array(img_baja, dtype=np.float32)
    img_hibrida = np.array(img_baja, dtype=np.float32)

    img_paso_alto=Gauss_kernel_Grey(img_alta.copy(),sigma_alto,0,0,0)
    img_paso_bajo=Gauss_kernel_Grey(img_baja.copy(),sigma_bajo,0,0,0) 
    
    img_paso_alto=img_alta - img_paso_alto
    
    img_hibrida = img_paso_alto + img_paso_bajo
    
    img_hibrida=Reescalar(img_hibrida,False,True)
    
    
    mi_lista.append(img_paso_alto)
    mi_lista.append(img_paso_bajo)
    mi_lista.append(img_hibrida)
    
    return mi_lista
    

#---------------------------------------------------------------
#-------------------BONUS---------------------------------------
#---------------------------------------------------------------

#Apartado 1

def f(x,sigma):
    t=exp(-0.5*(float)(pow(x,2)/pow(sigma,2)))
    return t

#Función que a partir de un sigma, crea una máscara
def calcular_kernel(sigma):
    tam=sigma*3+1
    mi_kernel=np.ones(sigma*6+1)
    
    for i in range(tam):
        mi_kernel[i]= sigma*3-i
        
    for i in range(tam,tam+sigma*3):
        mi_kernel[i]= i-3
        
    for i in range(len(mi_kernel)):
        mi_kernel[i]=f(mi_kernel[i],sigma)
        
    mi_kernel=mi_kernel/sum(mi_kernel)    
        
    return mi_kernel    



#Apartado 2

#Función que realiza una convolucion con un kernle y un vector
def calcular_convolucion(img_vector,kernel,tam_borde,tam_orig):

    
    vector_mod=np.zeros(len(img_vector))
    
    for i in range(tam_borde,tam_borde+tam_orig):
        vector_mod[i]= sum(img_vector[i-tam_borde:i-tam_borde+len(kernel)]*kernel)/len(kernel)
    
     
    return vector_mod  
    

#Función que añade un marco de un tamaño especificado a una imagen
def anadir_marco(img,tam_borde):
    
    img_mod=np.zeros((img.shape[0]+tam_borde*2,img.shape[1]+tam_borde*2))    
    
    for i in range(tam_borde,img.shape[0]+tam_borde):
        for j in range(tam_borde,img.shape[1]+tam_borde):
            img_mod[i,j]=img[i-tam_borde,j-tam_borde]
    
    return img_mod
    
#Apartado 3    

#Función que realiza una convolucion entre una imagen y un kernel
def mi_convolucion(img,sigma,colores=True):
    
    kernel=calcular_kernel(sigma)
    
    img_conv=np.array(img, dtype=np.float32)
    
    tam_origX=img.shape[1]
    tam_origY=img.shape[0]    
    
    if(colores):
        img_conv1=img_conv[:,:,0]
        img_conv2=img_conv[:,:,1]
        img_conv3=img_conv[:,:,2]
    else:
        img_conv1=img_conv
    
    img_conv1=anadir_marco(img_conv1,int(len(kernel)/2))
    if(colores):
        img_conv2=anadir_marco(img_conv2,int(len(kernel)/2))
        img_conv3=anadir_marco(img_conv3,int(len(kernel)/2))
    
  
               
    for i in range(img_conv1.shape[0]):
        img_conv1[i]=calcular_convolucion(img_conv1[i],kernel,int(len(kernel)/2),tam_origX)
        if(colores):
            img_conv2[i]=calcular_convolucion(img_conv2[i],kernel,int(len(kernel)/2),tam_origX)
            img_conv3[i]=calcular_convolucion(img_conv3[i],kernel,int(len(kernel)/2),tam_origX)

           
    for i in range(img_conv1.shape[1]):
        img_conv1[:,i]=calcular_convolucion(img_conv1[:,i],kernel,int(len(kernel)/2),tam_origY)
        if(colores):
            img_conv2[:,i]=calcular_convolucion(img_conv2[:,i],kernel,int(len(kernel)/2),tam_origY)
            img_conv3[:,i]=calcular_convolucion(img_conv3[:,i],kernel,int(len(kernel)/2),tam_origY)
    
    if(colores):               
        img_conv=cv2.merge((img_conv1,img_conv2,img_conv3))
    else:
        img_conv=img_conv1
        
    img_conv=Reescalar(img_conv,True)
                   
    
    return img_conv
    
#Apartado 4

#Función que quita los bordes de una imagen
def quitar_marcos(img,tam_marco):
    
    img=img[tam_marco:img.shape[0]-tam_marco,tam_marco:img.shape[1]-tam_marco]
    
    return img

#Función que calcula imagenes híbridas
def Mis_Imagenes_Hibridas(img_alta,img_baja,sigma_alto,sigma_bajo):
    
    mi_lista=[]

    img_paso_alto=np.zeros((img_alta.shape[0]+sigma_alto*6,img_alta.shape[1]+sigma_alto*6),dtype=np.float32) 
    img_paso_bajo=np.zeros((img_baja.shape[0]+sigma_bajo*6,img_baja.shape[1]+sigma_bajo*6),dtype=np.float32)
    
    
    img_paso_alto=mi_convolucion(img_alta.copy(),sigma_alto,False)
    img_paso_bajo=mi_convolucion(img_baja.copy(),sigma_bajo,False)
    
   
    img_paso_alto=quitar_marcos(img_paso_alto,sigma_alto*3)
    img_paso_bajo=quitar_marcos(img_paso_bajo,sigma_bajo*3)
    
       
    img_paso_alto=img_alta - img_paso_alto
    
    img_hibrida = img_paso_alto + img_paso_bajo
    
    img_hibrida=Reescalar(img_hibrida,False,True)
    
    mi_lista.append(img_paso_alto)
    mi_lista.append(img_paso_bajo)
    mi_lista.append(img_hibrida)
    
    return mi_lista

#Función que reduce a la mitad el tamaño de una imagen    
def reducir_img(img,colores=True):
    
    k=0
    t=0
    
    list_i=range(len(img))
    list_j=range(len(img[0]))
    list_i=[x for x in list_i if x%2 == 0]
    list_j=[x for x in list_j if x%2 == 0]
    
    if(colores):
        img_r=np.zeros((len(list_i),len(list_j),3))
    else:
        img_r=np.zeros((len(list_i),len(list_j)))
    
    for i in list_i:
        t=0
        for j in list_j:
            img_r[k,t]=img[i,j]
            t=t+1
        k=k+1
        
    return img_r
    
 
#Función que crea una piramide Gaussiana   
def mi_piramide_Gaussiana(img,sigma,n):
    
    lista_img_piramides = []

    for i in range(1,n+1):
        img=mi_convolucion(img,sigma,False)
        img=quitar_marcos(img,sigma*3)
        lista_img_piramides.append(img)
        img=reducir_img(img,False)

    return lista_img_piramides    
    



def main():
    
    #Apartado A
  
    mis_imagenes = ["imagenes/bicycle.bmp","imagenes/bird.bmp","imagenes/cat.bmp","imagenes/dog.bmp","imagenes/einstein.bmp","imagenes/fish.bmp","imagenes/marilyn.bmp","imagenes/motorcycle.bmp","imagenes/plane.bmp","imagenes/submarine.bmp"]    
    lista_imagenes = []
    
    bordes=[cv2.BORDER_WRAP,cv2.BORDER_WRAP,cv2.BORDER_CONSTANT,cv2.BORDER_CONSTANT]
    nombre_bordes=['WRAP','WRAP','Constante','Constante']
    
    for image in mis_imagenes:
        lista_imagenes.append(cv2.imread(image,0))
        
    imprimir(lista_imagenes,4,4,mis_imagenes,True)
    
    
    #Apartado B
    
    lista_imagenesB = []
    titulos=[]
    
    for i in range(1,5):
        mi_img=Gauss_alisamiento(lista_imagenes[5],i*2,i*2,bordes[i-1])
        lista_imagenesB.append(np.array(mi_img, dtype=np.uint8))
        titulos.append("Smooth Gauss sigma=" + str(i*2) + " con borde " + nombre_bordes[i-1])    

    imprimir(lista_imagenesB,2,2,titulos,True)
    
    
    #Apartado C
    
    lista_imagenesB = []
    titulos = []
    
    for i in range(1,5):
        mi_img=Gauss_kernel_Grey(lista_imagenes[5].copy(),i,0,0,bordes[i-1])
        lista_imagenesB.append(np.array(mi_img, dtype=np.uint8))
        titulos.append("Smooth con kernels sigma=" + str(i) + " con borde " + nombre_bordes[i-1])    
        
    imprimir(lista_imagenesB,2,2,titulos,True) 
    
    #Apartado D
    
    lista_imagenesB = []
    titulos=[]
    var=["x","y"]
    sigmas=[1,1,2,2]
    
    for i in range(1,5):
        mi_img=kernel_derivada(lista_imagenes[6],1,1,i%2,bordes[i-1],sigmas[i-1])
        lista_imagenesB.append(np.array(mi_img, dtype=np.uint8))
        titulos.append("Primera derivada '" + var[i%2] +"' con sigma " + str(sigmas[i-1]) + " con borde " + nombre_bordes[i-1])    
    
    imprimir(lista_imagenesB,2,2,titulos,True)  

    #Apartado E
    
    lista_imagenesB = []
    titulos=[]
    
    
    for i in range(1,5):
        mi_img=kernel_derivada(lista_imagenes[6],2,2,i%2,bordes[i-1],sigmas[i-1])
        lista_imagenesB.append(np.array(mi_img, dtype=np.uint8))
        titulos.append("Segunda derivada '" + var[i%2] +"' con sigma " + str(sigmas[i-1]) + " con borde " + nombre_bordes[i-1])    
    
    imprimir(lista_imagenesB,2,2,titulos,True)
    
    #Apartado F
    
    lista_imagenesB = []
    titulos=[]
    
    for i in range(1,3):
        mi_img=Laplaciana_Gaussiana(lista_imagenes[0],bordes[i],i)
        lista_imagenesB.append(np.array(mi_img, dtype=np.uint8))
        titulos.append("Laplaciana-Gaussiana con sigma " + str(i) + " y borde " + nombre_bordes[i])    
    
    imprimir(lista_imagenesB,2,1,titulos,True)
    
    #Apartado G
    
    
    mi_img=cv2.imread("imagenes/bicycle.bmp",0)    
    lista_img_piramides= piramide_Gaussiana(mi_img,bordes[2],5) 
         
    mi_lista=[]
    titulos=[]
    mi_lista.append(construir_piramide(lista_img_piramides))
    titulos.append("Piramide Gaussiana con borde " + nombre_bordes[2])   
    
    imprimir(mi_lista,1,1,titulos,True)
    
    #Apartado H
    
    mi_img=cv2.imread("imagenes/marilyn.bmp",0)    
    lista_img_piramides= piramide_Laplaciana(mi_img,bordes[2],5) 
       
        
    mi_lista=[]
    titulos=[]
    mi_lista.append(construir_piramide(lista_img_piramides))
    titulos.append("Piramide Laplaciana con borde " + nombre_bordes[2])
    
    imprimir(mi_lista,1,1,titulos,True)
    
    #Ejercicio 2: Imágenes Hibridas
    
    img_bicycle=cv2.imread("imagenes/bicycle.bmp",0)
    img_motorcycle=cv2.imread("imagenes/motorcycle.bmp",0)
    img_fish=cv2.imread("imagenes/fish.bmp",0)
    img_submarine=cv2.imread("imagenes/submarine.bmp",0)
    img_einstein=cv2.imread("imagenes/einstein.bmp",0)
    img_marilyn=cv2.imread("imagenes/marilyn.bmp",0)
    
    mi_lista1=Get_Imagenes_Hibridas(img_bicycle,img_motorcycle,5,7)
    mi_lista2=Get_Imagenes_Hibridas(img_fish,img_submarine,5,7)
    mi_lista3=Get_Imagenes_Hibridas(img_einstein,img_marilyn,5,7)
    
    mi_lista=mi_lista1+mi_lista2+mi_lista3
    titulos=["Paso_alto","Paso_bajo","Hibrida","Paso_alto","Paso_bajo","Hibrida","Paso_alto","Paso_bajo","Hibrida"]
    imprimir(mi_lista,3,3,titulos,True)
       
    mi_img=mi_lista3[2]    
    lista_img_piramides= piramide_Gaussiana(mi_img,0,5) 
       
        
    mi_lista=[]   
    mi_lista.append(construir_piramide(lista_img_piramides))
    titulos=["Piramide Gaussiana con imagen Hibrida"]
    
    imprimir(mi_lista,1,1,titulos,True)
    
    #----------------BONUS---------------------------------
    
    #Apartado 3
    
    lista_imagenesB = []
    titulos = []
    
    img=cv2.imread("imagenes/fish.bmp")
    
    for i in range(1,3):
        mi_img=mi_convolucion(img.copy(),i*2)
        lista_imagenesB.append(np.array(mi_img, dtype=np.uint8))
        titulos.append("Mi convolucion sigma=" + str(i*2))    
    
    imprimir(lista_imagenesB,2,1,titulos)
    
    
    #Apartado 4
    
    img_bicycle=cv2.imread("imagenes/bicycle.bmp",0)
    img_motorcycle=cv2.imread("imagenes/motorcycle.bmp",0)
    img_fish=cv2.imread("imagenes/fish.bmp",0)
    img_submarine=cv2.imread("imagenes/submarine.bmp",0)
    img_einstein=cv2.imread("imagenes/einstein.bmp",0)
    img_marilyn=cv2.imread("imagenes/marilyn.bmp",0)
    
    
    mi_lista=Mis_Imagenes_Hibridas(img_bicycle.copy(),img_motorcycle.copy(),5,7)
    
    
    mi_img=mi_lista[2]    
    lista_img_piramides=mi_piramide_Gaussiana(mi_img,2,5) 
    
    mi_piramide = construir_piramide(lista_img_piramides)
    
    titulo=["Imagen hibrida en piramide Gaussiana"]
    mi_lista=[]
    mi_lista.append(mi_piramide)
    
    imprimir(mi_lista,1,1,titulo,True)
    
    

if __name__ == "__main__": 
    main()   
    