from numpy.fft import fft,ifft
import numpy as np
import matplotlib.pyplot as plt

f1 = 250 # fréquence de la fonction (=fréquence fondamentale)
T = 1/f1 # période de la fonction
w1 = 2*np.pi*f1 # pulsation fondamentale

 # On définit ci-dessous quelques fonctions à décomposer ensuite en série de Fourier.
def creneau(t):
    "Fonction créneau ; valide pour tout t"
    return (-1)**np.floor(2*t/T) # ici c'est une fonction créneau

def triangle(t):
    "Fonction triangle ; valide pour tout t"
    return (-1)**np.floor(2*t/T)*(4*t/T-(1+2*np.floor(2*t/T))) # et là unefonction triangle !

def fonctionbizarre(t):
    "Fonction 'bizarre' ; n'est définie correctement que sur une période, de 0 à T"
    if (t<T/3):
        return 6/T*np.sqrt(T**2/36-(t-T/6)**2)
    elif (t<3*T/4):
        return 24/5/T*(t-T/3)
    else:
        return -8/T*(t-T)
    
fonctionbizarrev = np.vectorize(fonctionbizarre) # On la vectorise

 # On échantillonne le temps sur une période de u(t)
N = 1024 # doit absolument être pair et, de préférence, une puissance de 2 (pour le fonctionnement de fft)
te = T/N # pas d'échantillonnage
fe=1/te # fréquence d'échantillonnage
temps = np.linspace(0,T,num=N,endpoint=False)

u = creneau # On choisit ici la fonction à décomposer : creneau, triangle, ou autre...

echantillons = u(temps) # sur une seule période

 # Tracé des échantillons sur quatre périodes
tempss = np.linspace(-T,3*T,num=4*N) # ne sert QUE au tracé de u(t) sur quatre périodes
echantillonss = np.concatenate((echantillons,echantillons,echantillons,echantillons)) # idem
zeros = np.zeros_like(tempss)
plt.figure(figsize=(8,4))
plt.plot(tempss,echantillonss,'r-')
plt.plot(tempss,zeros,'k-')
plt.xlabel('t(s)')
plt.ylabel('u(t)')
#plt.savefig('bizarre.png')
plt.show()

 # Calcul de la transformée de Fourier discrète et des fréquences
tfd = fft(echantillons)/N
freq = np.linspace(0,N/T,num=N,endpoint=False)

# Représentation graphique du module de la tfd en fonction de la fréquence.
spectre =np.absolute(tfd)*2 # attention le s0 n'est pas à multiplier par deux
spectre[0]=spectre[0]/2
plt.figure(figsize=(8,4))
plt.vlines(freq[0:50],[0],spectre[0:50],'r') # on ne dessine que les 50 premiers harmoniques
# Le "problème" est bien sûr celui du repliement de spectre, pour les fonctions qui y sont sujettes, c'est-à-dire la plupart... Les coefficients les plus affectés
#par ce problème sont ceux des rangs les plus élevés en général, donc on ne les
#dessine pas.
plt.xlabel('f(Hz)')
plt.ylabel('')
plt.title('Spectre de u(t)')
plt.axis([-100,50*w1/2/np.pi,0,spectre.max()*1.1]) # le -100 est là pour que le pic de fréquence nulle soit visible
plt.grid()
#plt.savefig('bizarrespectre.png')
plt.show()
