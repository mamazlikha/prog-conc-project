#cell 13
#!/usr/bin/python
import threading
import time

# thread function to start

def print_function():
    print("Hello world!\n")


x = threading.Thread(target=print_function, ) # creating and initializing thread
x.start() 

#cell 16
import threading
import time

N = 10;

def thread_function(i):
    time.sleep(2) #Necessaire à la mise en evidence
    print(i, sep='', end='')


print("Lancement de ", N ," threads :")
    
i=0
while i<N :
    i=i+1
    x = threading.Thread(target=thread_function, args=(i,))
    x.start()

#cell 18
import threading
import time

N = 1000

left = 0
right = 0
total = 0


def incr_function(leftOrRight):
    global total
    global right
    global left
    i=0
    while i<N :
        i=i+1
        save = total
        print('',end='');
        if leftOrRight :
            left = left+1
        else :
            right = right+1
        total = save +1

print("Lancement de ", N ," threads :")
x = threading.Thread(target=incr_function, args=(0,))
y = threading.Thread(target=incr_function, args=(1,))
z = threading.Thread(target=incr_function, args=(0,))
t = threading.Thread(target=incr_function, args=(1,))
x.start()
y.start()
z.start()
t.start()
x.join()
y.join()
z.join()
t.join()
print("total expected =",N+N+N+N)
print("total got =", total)

#cell 20
import threading
import time

N = 1000

left = 0
right = 0
total = 0

verrou = threading.Lock()


def incr_function(leftOrRight):
    global total
    global right
    global left
    i=0
    while i<N :
        verrou.acquire()
        i=i+1
        save = total
        print('',end='');
        if leftOrRight :
            left = left+1
        else :
            right = right+1
        total = save +1
        verrou.release()


print("Lancement de ", 4 ," threads :")
x = threading.Thread(target=incr_function, args=(0,))
y = threading.Thread(target=incr_function, args=(1,))
z = threading.Thread(target=incr_function, args=(0,))
t = threading.Thread(target=incr_function, args=(1,))
x.start()
y.start()
z.start()
t.start()
x.join()
y.join()
z.join()
t.join()
print("total expected =",N+N+N+N)
print("total got =", total)

#cell 22
# -*- coding: utf-8 -*-
""" rwlock.py
    A class to implement read-write locks on top of the standard threading
    library.
    This is implemented with two mutexes (threading.Lock instances) as per this
    wikipedia pseudocode:
    https://en.wikipedia.org/wiki/Readers%E2%80%93writer_lock#Using_two_mutexes
    Code written by Tyler Neylon at Unbox Research.
    This file is public domain.
"""


# _______________________________________________________________________
# Imports

from contextlib import contextmanager
from threading  import Lock


# _______________________________________________________________________
# Class

class RWLock(object):
    """ RWLock class; this is meant to allow an object to be read from by
        multiple threads, but only written to by a single thread at a time. See:
        https://en.wikipedia.org/wiki/Readers%E2%80%93writer_lock
        Usage:
            from rwlock import RWLock
            my_obj_rwlock = RWLock()
            # When reading from my_obj:
            with my_obj_rwlock.r_locked():
                do_read_only_things_with(my_obj)
            # When writing to my_obj:
            with my_obj_rwlock.w_locked():
                mutate(my_obj)
    """

    def __init__(self):

        self.w_lock = Lock()
        self.num_r_lock = Lock()
        self.num_r = 0

    # ___________________________________________________________________
    # Reading methods.

    def r_acquire(self):
        self.num_r_lock.acquire()
        self.num_r += 1
        if self.num_r == 1:
            self.w_lock.acquire()
        self.num_r_lock.release()

    def r_release(self):
        assert self.num_r > 0
        self.num_r_lock.acquire()
        self.num_r -= 1
        if self.num_r == 0:
            self.w_lock.release()
        self.num_r_lock.release()

    @contextmanager
    def r_locked(self):
        """ This method is designed to be used via the `with` statement. """
        try:
            self.r_acquire()
            yield
        finally:
            self.r_release()

    # ___________________________________________________________________
    # Writing methods.

    def w_acquire(self):
        self.w_lock.acquire()

    def w_release(self):
        self.w_lock.release()

    @contextmanager
    def w_locked(self):
        """ This method is designed to be used via the `with` statement. """
        try:
            self.w_acquire()
            yield
        finally:
            self.w_release()

#cell 24
import threading

sem = threading.Semaphore(1)

def fun1():
    print("fun1 demarre")
    sem.acquire()
    for loop in range(1,4):
        print("fun1 en cours ... {}".format(loop))

    sem.release()
    print("fun1 terminée")


def fun2():
    print("fun2 demarre")
    while not sem.acquire(blocking=False): # check tant qu'on ne peut pas recuperer le semapghore du t1 
        print("Fun2 Pas de Semaphore dispo") # on fait un affichage tant que la semaphore n'est pas libérer par la fun1() si 
        # fun()1 prend la semaphore avant fun2(). Dans l'autre cas on n'aura pas à entrer dans cette boucle quasi-infinie
        
    else:
        print("\n fun2() recoit Semphore \n")
        for loop in range(1, 4): # une loop pour voir la progression de cette fonction
            print("Fun2 en cours ... {}".format(loop))
            
    sem.release()

t1 = threading.Thread(target = fun1)
t2 = threading.Thread(target = fun2)
t1.start()
t2.start()



#cell 27
import threading

# Class monitor pour gérer un parking qui a une capacité, capacity
# Nous avons pris l'exemple du TD qu'on a fait en cours.
# Selon la variable freePlaces on bloquera la fonction tackPlace ou freePlace
# Par exemple, la fonction tackPlace ne pourra pas incrémenter la variable freePlaces tant qu'il n'y a pas
# de place libre, donc elle va faire wait(). Quand la fonction freePlace, décrémente cette variable, elle notifiera le thread
# ayant executé la fonction takePlace afin qu'il continue son execution.

class CarParkingMonitor :
    def __init__(self, capacity) :
        self.capacity = capacity
        
        self.freePlaces = capacity
        self.useLock = threading.Lock() 
        self.useAvailable = threading.Condition(self.useLock)
    
    ## prend un place libre ou attend qu'une se libère
    def takePlace(self) :
        with self.useLock :
            while self.freePlaces <=0 :
                self.useAvailable.wait()
            self.freePlaces=self.freePlaces-1
            self.useAvailable.notifyAll()
        
    ## prend une voiture ou attend qu'une soit disponible
    def freePlace(self) :
        with self.useLock :
            while self.freePlaces >= self.capacity :
                self.useAvailable.wait()
            self.freePlaces=self.freePlaces+1
            self.useAvailable.notifyAll()

monitor = CarParkingMonitor(3)
## permet au différent appel à la fonction print() de ne pas entrer en concurrence
printLock = threading.Lock()

def whoArrive(i) :
    global monitor
    global printLock
    
    printLock.acquire()
    if monitor.freePlaces==0 : 
        print(i,"is waiting to park")
    printLock.release()
    
    monitor.takePlace()
    
    printLock.acquire()
    print(i,"parked")
    printLock.release()
    
    
def whoLeave(i) :
    global monitor
    global printLock
    
    printLock.acquire()
    if monitor.freePlaces!=0 : 
        print(i,"is waiting to leave")
    printLock.release()
    
    monitor.freePlace()
    
    printLock.acquire()
    print(i,"leaved")
    printLock.release()

# creation de 10 "personne" qui veulent rendre un véhicule
for i in range(1,10) :
    t1 = threading.Thread(target = whoArrive, args=(i,))
    t1.start()

# creation de 10 "personne" qui veulent emprunter un véhicule
for i in range(1,10) :
    t2 = threading.Thread(target = whoLeave, args=(10+i,))
    t2.start()

#cell 29
def is_running_from_ipython():
    # Renvoie True si on execute le code depuis un notebook
       from IPython import get_ipython
       return get_ipython() is not None

# Permet de savoir qui exécute le code
import platform
MICHEL = ('michel' in platform.node()) or ('charvin' in platform.node())

#cell 30
class Debug:
    def __init__(self, debug=False):
        self.debug = debug
        
    def affiche(self, *args):
        if self.debug:
            print(args)
            
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

#cell 31
# Différents imports
import copy
import math
import time
import numpy as np

%matplotlib inline
import matplotlib.pyplot as plt

#cell 32
# Reglage de l'affichage

# Définition des affichage possible définition
try :
    import affichage
except :
    !echo 'Aucun = 0' >affichage.py
    !echo 'tkinter = 1' >>affichage.py # Not implemented
    !echo 'OpenCV = 2' >> affichage.py # Not completly implemented

    !echo 'PERSONNE = (0.0, 0.0, 1.0)' >>affichage.py
    !echo 'IMMOBILE = (1.0, 0.0, 0.0)' >>affichage.py
    !echo 'OBSTACLE = (0.0, 1.0, 0.0)' >>affichage.py
    !echo 'EMPTY = (1.0, 1.0, 1.0)' >>affichage.py
    !echo 'ARRIVEE = (0.0, 0.0, 0.0)' >>affichage.py # Not used
    import affichage
    
import cv2

UI = affichage.OpenCV

#cell 33
if MICHEL:
    petite_grille = "maps/map-initial.txt" # fichier décrivant une petite grille
    grande_grille = "maps/map-final.txt"  # fichier décrivant une grande grille
    debug = Debug(True)
    
petite_grille = "maps/map-initial.txt" # fichier décrivant une petite grille
grande_grille = "maps/map-final.txt"  # fichier décrivant une grande grille
debug = Debug(True)

#cell 34
# Constant définition
try :
    import const
except: 
    !echo 'EMPTY = 0' >const.py
    !echo 'OBSTACLE = 1' >>const.py
    !echo 'PERSONNE = 2' >> const.py
    !echo 'IMMOBILE = 3' >> const.py
    !echo 'ARRIVEE = 4' >> const.py
    import const

#cell 35
# Lecture du fichier de configuration
def open_file(filename, debug):
    debug.affiche("Ouverture du fichier", filename)
    fileMap = open(filename, 'r')
    return fileMap

file = open_file(petite_grille, debug)

#cell 37
def read_grid_size(file, debug):
    line = file.readline()
    while line[0]=='#':	# on saute les commentaires
        line = file.readline()
    
    # Lecture de la taille de la grille
    while line[0]!='#':	# on lit jusqu'au prochain commentaires
        if (line[0]=='X'):
            X = int(line.split(' ')[1])
        if (line[0]=='Y'):
            Y = int(line.split(' ')[1])
        line = file.readline()

    debug.affiche("Simulation pour une grille:", X, "*", Y)
    
    return X, Y

#cell 38
X, Y = read_grid_size(file, debug)

#cell 39
# Definition de la grille
class Grille:
    # Crée un objet list
    def __init__(self, Xmax, Ymax, UI, debug):
        debug.affiche("Initilisation de l'objet grille")
        self.Xmax = Xmax
        self.Ymax = Ymax
        if const.EMPTY==0:
            self.map = np.zeros((Xmax, Ymax))
        elif const.EMPTY==1:
            self.map = np.ones((Xmax, Ymax))
        else:
            self.map = np.array(Xmax, Ymax)
            for i in range(Xmax):
                for j in range(Ymax):
                    self.map[i,j] = const.EMPTY
        self.mapList = []
        self.obstacleList = []
        self.personneList = []
        self.debug = debug
        self.UI = UI

    def SetObstacle(self, obstacle):
        for x in range(obstacle.Xmin, obstacle.Xmax+1):
            for y in range(obstacle.Ymin, obstacle.Ymax+1):
                self.map[x][y] = const.OBSTACLE
        self.obstacleList += [obstacle]

    def SetPersonne(self, personne):
        if self.map[personne.X][personne.Y]==const.EMPTY:
            # la case est vide
            self.map[personne.X][personne.Y] = const.PERSONNE
            self.personneList += [personne]
            return True
        else:
            debug.affiche("Personne non créée, la place est occupée", personne.X, personne.Y)
            return False
        
    def removePersonne(self,personne):
        self.personneList.remove(personne)
        
    def isCellEmpty(self, x, y):
        return self.map[x][y]==const.EMPTY
    
    def Set(self, x, y, status):
        self.map[x][y] = status
        
    def Save(self, name):
        with open(name, 'w') as file:
            file.write("# taille de la grille\n")
            file.write("X "+str(self.Xmax)+"\n")
            file.write("Y "+str(self.Ymax)+"\n")
            file.write("# position des obstacles\n")
            file.write("# Xmin Xmax Ymin Ymax\n")
            file.write("# si pas obstacle : mettre -1 comme valeur\n")
            file.write("# "+str(len(self.obstacleList))+" obstacles\n")
            for o in self.obstacleList:
                file.write(str(o.Xmin)+" "+str(o.Xmax)+" "+str(o.Ymin)+" "+str(o.Ymax)+"\n")
            file.write("# position initiale des personnes (X, Y) et case de sortie (X, Y)\n")
            file.write("# "+str(len(self.personneList))+" personnes\n")
            for p in self.personneList:
                file.write(str(p.X)+" "+str(p.Y)+" "+str(p.Xsortie)+" "+str(p.Ysortie)+"\n")
                
    # Solution imparfaite car une personne peut être bloquée par un obstacle
    def Simulation(self):
        if self.debug.debug or self.UI==affichage.OpenCV:
            self.mapList.append(copy.copy(self.map))
        if self.UI==affichage.OpenCV:
            cv2.imshow("img", createImage(self.map))
            cv2.waitKey(500)
    
        while len(self.personneList)>0: # il y a encore une personne dans la liste qui peut bouger
            for personne in self.personneList:
                end, immobile = personne.deplacePersonne(self, self.debug)
                if end:
                    self.debug.affiche("Une personne vient de sortir, il reste", len(self.personneList), "personnes")
            if self.debug.debug or self.UI==affichage.OpenCV:
                self.mapList.append(copy.copy(self.map))
                if self.UI==affichage.OpenCV:
                    cv2.imshow("img", createImage(self.map))
                    cv2.waitKey(500)
    
        if self.UI==affichage.OpenCV:
            cv2.destroyAllWindows()

#cell 40
# Creation de la grille
grid = Grille(X, Y, affichage.Aucun, debug)

#cell 42
class Obstacle():
    def __init__(self, grid,  Xmin, Xmax, Ymin, Ymax, debug):
        self.Xmin = Xmin
        self.Xmax = Xmax
        self.Ymin = Ymin
        self.Ymax = Ymax
        self.debug = debug
        grid.SetObstacle(self)
        debug.affiche("Création de l'obstacle (", Xmin, Xmax, ") - (", Ymin, Ymax, ")")

#cell 43
def read_obstacles(file, grid, Obstacle, debug):
    # Lecture de la liste des obstacles et placement sur la grille
    line = file.readline()
    while line[0]=='#':	# on saute les commentaires
        line = file.readline()

    while line[0]!='#':	# on lit jusqu'au prochain commentaires
        coordonnee = line.strip().split(' ')
        if int(coordonnee[0])>=0:
            assert (int(coordonnee[0])>0 and int(coordonnee[2])>0)
            assert (int(coordonnee[1])<X)
            assert (int(coordonnee[3])<Y)
            Obstacle(grid, int(coordonnee[0]), int(coordonnee[1]), int(coordonnee[2]), int(coordonnee[3]), debug)
        else:
            debug.affiche("obstacle non créé", coordonnee)
        line = file.readline()        

    debug.affiche(len(grid.obstacleList), "obstacles ont été créés")

#cell 44
# On lit tous les obstacles contenus dans le fichier de configuration
read_obstacles(file, grid, Obstacle, debug)

#cell 45
# Affichage de la grille
plt.imshow(grid.map, cmap='Paired', vmin=0, vmax=5)
plt.show()

#cell 47
def distance(X, Y):
    return int(np.abs(X[0]-Y[0])+np.abs(X[1]-Y[1]))

class Personne():
    def __init__(self, grid, Xinit, Yinit, Xsortie, Ysortie, debug):
        self.X = Xinit
        self.Y = Yinit
        self.Xsortie = Xsortie
        self.Ysortie = Ysortie
        self.debug = debug
        personneCrée = grid.SetPersonne(self)
        if personneCrée:
            debug.affiche("Création de la personne qui part de", Xinit, Yinit, "pour aller à", Xsortie, Ysortie)
    
    def getAvailableMove(self, grid): # retourne la liste triée des déplacements possibles
        # distance à la grille
        dist_actuel = distance((self.X, self.Y), (self.Xsortie, self.Ysortie))
        depl = []
        # on calcule la distance entre les case adjacente et la sortie
        for x in range(self.X-1, self.X+2):
            for y in range (self.Y-1, self.Y+2):
                if (x!=self.X or y!=self.Y) and (x==self.X or y==self.Y) and (x>=0 and y>=0 and x<grid.Xmax and y<grid.Ymax): # deplacement uniquement points cardinaux
                    # on est sur une case adjacente et on est sur la grille
                    if grid.isCellEmpty(x, y): # si la case est vide
                        dist = distance((self.Xsortie, self.Ysortie), (x,y))
                        if (dist<dist_actuel): # on se rappproche
                             depl.append((dist, x, y))
        return sorted(depl)

    def getBetterMove(self, grid):
        return self.getAvailableMove(grid)[0]

    def Move(self, grid, depl):
        debug.affiche("Déplacement de la personne de (", self.X, self.Y, ") à (", depl[1], depl[2], ")")
        grid.Set(self.X, self.Y, const.EMPTY)
        self.X = depl[1]
        self.Y = depl[2]
        grid.Set(self.X, self.Y, const.PERSONNE)
        
    def deplacePersonne(self, grid, debug):
        if self.X==self.Xsortie and self.Y==self.Ysortie:
            # la personne est arrivée à la sortie
            # on retire la personne de la liste
            grid.removePersonne(self)
            # on libere la case de sortie
            grid.Set(self.X, self.Y, const.EMPTY)
            return True, False
    
        # sinon on recherche les déplacements possibles
        availableMove = self.getAvailableMove(grid)
        debug.affiche("La personne (", self.X, self.Y, ") peut se déplacer sur :", availableMove)
            
        if len(availableMove)>0:
            # on se déplace sure la nouvelle case
            self.Move(grid, availableMove[0]) 
            return False, False
        
        # sinon aucun d"éplacement n'est possible
        if grid.UI==affichage.OpenCV:
            grid.Set(self.X, self.Y, const.IMMOBILE)
        debug.affiche("La personne (", self.X, self.Y, "ne peut pas se déplacer")
        return False, True

#cell 48
def read_personnes(file, grid, Personne, nombre, debug):
    # Lecture de la liste des personnes et placement sur la grille
    # s'arrête quand le nombre de personnes souhaité est atteind
    # Si nombre = -1, on lit jusqu'à la fin du fichier (idem si nombre est supérieur aux nombres de personnes dans le fichier)
    line = file.readline()
    while line[0]=='#':	# on saute les commentaires
        line = file.readline()
        
    while line:	# on lit jusqu'à la fin du fichier
        coordonnee = line.strip().split(' ')
        if coordonnee[0]!='#':
            assert (int(coordonnee[0])>0 and int(coordonnee[1])>0
                and int(coordonnee[2])<X or int(coordonnee[3])<Y) # La personne est sur la grille
            assert ((int(coordonnee[2])==0 or int(coordonnee[2])==X-1)  # elle sort sur un bord X
                and                                                  # et
                (int(coordonnee[3])==0 or int(coordonnee[3])==Y-1)) # elle sort sur un bord Y
            Personne(grid, int(coordonnee[0]), int(coordonnee[1]), int(coordonnee[2]), int(coordonnee[3]), debug)
            
        if len(grid.personneList)==nombre:
            debug.affiche(len(grid.personneList), "personnes ont été créées")
            file.close()
            return
        line = file.readline()
    debug.affiche("Fermeture du fichier")
    file.close()
    
    debug.affiche("On a lut tout le fichier,", len(grid.personneList), "personnes ont été créées")

#cell 49
# On lit toutes les personnes contenues dans le fichier de configuration
read_personnes(file, grid, Personne, -1, debug)

#cell 50
# Affichage de la grille
plt.imshow(grid.map, cmap='Paired', vmin=0, vmax=5)
plt.show()

#cell 52
# Uniquement nécessaire pour la visualisation en OpenCV
def createImage(image):
    X = image.shape[0]
    Y = image.shape[1]
    ZOOM = max(min(1024//X, 1024//Y), 1)
    img = np.zeros((X*ZOOM, Y*ZOOM, 3), dtype=np.float)
    for i in range(X):
        for t in range(ZOOM):
            for j in range(Y):
                for u in range(ZOOM):
                    if image[i][j]==const.EMPTY:
                        img[ZOOM*i+t][ZOOM*j+u]=affichage.EMPTY
                    elif image[i][j]==const.OBSTACLE:
                        img[ZOOM*i+t][ZOOM*j+u]=affichage.OBSTACLE
                    elif image[i][j]==const.PERSONNE:
                        img[ZOOM*i+t][ZOOM*j+u]=affichage.PERSONNE
                    elif image[i][j]==const.IMMOBILE:
                        img[ZOOM*i+t][ZOOM*j+u]=affichage.IMMOBILE
                    elif image[i][j]==const.ARRIVEE:  
                        img[ZOOM*i+t][ZOOM*j+u]=affichage.ARRIVEE
    return img

#cell 53
# Affichage en OpenCV de la grille créée
if UI==affichage.OpenCV:
    cv2.imshow("img", createImage(grid.map))
    cv2.waitKey(5000) # Affichage pendant 5 secondes
    cv2.destroyAllWindows()

#cell 55
# Création d'une grille de 1024*1024
debug = Debug(True)
UI = affichage.Aucun

X = 1024
Y = 1024
grid = Grille(X, Y, UI, debug)

#cell 56
# La grille contient 2^4 obstacles
nb_obs = np.power(2,4)

coordonnee = np.zeros((4,1), dtype=np.int)
for i in range(nb_obs):
    for j in range(4):
        val_min = 1+50*i+np.random.randint(10)*10*j
        if j%2 == 0:
            val_max = min(1+val_min+np.random.randint(20), grid.Xmax)
        else:
            val_max = min(1+val_min+np.random.randint(20), grid.Ymax)
        coordonnee[j] = np.random.randint(val_min, val_max)
    
    assert (int(coordonnee[0])>0 and int(coordonnee[2])>0)
    assert (int(coordonnee[1])<X)
    assert (int(coordonnee[3])<Y)
    Obstacle(grid, int(coordonnee[0]), int(coordonnee[1]), int(coordonnee[2]), int(coordonnee[3]), debug)
    
debug.affiche(len(grid.obstacleList), "obstacles ont été créés")

#cell 57
# La grille contient 2^10 personnes
nb_person = np.power(2,10)

coordonnee = np.zeros((2,1), dtype=np.int)
sortie = np.zeros((2,1), dtype=np.int)
while len(grid.personneList)<nb_person:
    coordonnee[0] = np.random.randint(1,X)
    coordonnee[1] = np.random.randint(1,Y)
    sortie[0] = (X-1)*np.random.randint(2)
    sortie[1] = (Y-1)*np.random.randint(2)
    Personne(grid, int(coordonnee[0]), int(coordonnee[1]), int(sortie[0]), int(sortie[1]), debug)
    
debug.affiche(len(grid.personneList), "personnes ont été créées")

#cell 58
# Sauvegarde de l'objet grille
grid.Save(grande_grille)

#cell 59
# Affichage de la grille créée
if UI==affichage.OpenCV:
    cv2.imshow("img", createImage(grid.map))
    cv2.waitKey(5000) # Affichage pendant 5 secondes
    cv2.destroyAllWindows()

#cell 62
# En mode debug avec 1 personne, sans UI
debug = Debug(True)

file = open_file(petite_grille, debug)  # On ouvre le fichier
X, Y = read_grid_size(file, debug) # On lit la taille
grid = Grille(X, Y, affichage.Aucun, debug)     # On créée l'objet grille
read_obstacles(file, grid, Obstacle, debug)  # On lit et positionne les obstacles
read_personnes(file, grid, Personne, 1, debug) # On lit et positionne une seule personne
print("*"*20)
print("Début de la simulation")
grid.Simulation()

#cell 64
for img in grid.mapList:
    plt.imshow(img, cmap='Paired', vmin=0, vmax=5)
    plt.show()

#cell 66
if UI==affichage.OpenCV:
    for image in grid.mapList:
        cv2.imshow("img", createImage(image))
        cv2.waitKey(500) # Wait for 0.5 second
    cv2.destroyAllWindows()

#cell 68
# En mode sans trace avec 1 personne mais avec UI
if UI==affichage.OpenCV:
    debug = Debug(False)
    file = open_file(petite_grille, debug)  # On ouvre le fichier
    X, Y = read_grid_size(file, debug) # On lit la taille
    grid = Grille(X, Y, affichage.OpenCV, debug)     # On créée l'objet grille
    read_obstacles(file, grid, Obstacle, debug)  # On lit et positionne les obstacles
    read_personnes(file, grid, Personne, 1, debug) # On lit et positionne une seule personne
    grid.Simulation()

#cell 75
class Grille_1(Grille):
    waitersCount = 0
    
    # Crée un objet list
    def __init__(self, Xmax, Ymax, UI, debug):
        Grille.__init__(self, Xmax, Ymax, UI, debug)
    
    def Simulation(self):
        if self.debug.debug or self.UI==affichage.OpenCV:
            self.mapList.append(copy.copy(self.map))
        if self.UI==affichage.OpenCV:
            cv2.imshow("img", createImage(self.map))
            cv2.waitKey(500)
            
        moveMade = True
        while moveMade : # il y a encore une personne dans la liste qui peut bouger
            moveMade = False;
            self.waitersCount = 0
            for personne in self.personneList:
                end, immobile = personne.deplacePersonne(self, self.debug)
                if immobile==False:
                    moveMade = True
                else :
                    self.waitersCount+=1
                if end:
                    self.debug.affiche("Une personne vient de sortir, il reste", len(self.personneList), "personnes")
            if self.debug.debug or self.UI==affichage.OpenCV:
                self.mapList.append(copy.copy(self.map))
                if self.UI==affichage.OpenCV:
                    cv2.imshow("img", createImage(self.map))
                    cv2.waitKey(500)
    
        if self.UI==affichage.OpenCV:
            cv2.destroyAllWindows()

#cell 76
# En mode debug avec 1 personne, sans UI
debug = Debug(True)

file = open_file(petite_grille, debug)  # On ouvre le fichier
X, Y = read_grid_size(file, debug) # On lit la taille
grid = Grille_1(X, Y, affichage.Aucun, debug)     # On créée l'objet grille
read_obstacles(file, grid, Obstacle, debug)  # On lit et positionne les obstacles
read_personnes(file, grid, Personne, -1, debug) # On lit et positionne une seule personne
print("*"*20)
print("Début de la simulation")
grid.Simulation()

#cell 77
for img in grid.mapList:
    plt.imshow(img, cmap='Paired', vmin=0, vmax=5)
    plt.show()

#cell 78
''' plot du temps d'exécution en fonction du nombre de personnes '''

debug = Debug(False)

duration_1 = []  # i est le numéro du scénario
nb_personnes = []
for i in range(11):
    nb_personnes += [np.power(2, i)]
    print("*"*30)
    print("pour", nb_personnes[-1], "personnes")

    file = open_file(grande_grille, debug)  # On ouvre le fichier
    X, Y = read_grid_size(file, debug) # On lit la taille
    debug.affiche("simulation pour une grille:", X, Y)
    grid = Grille_1(X, Y,affichage.Aucun, debug)     # On créée l'objet grille
    read_obstacles(file, grid, Obstacle, debug)  # On lit et positionne les obstacles
    read_personnes(file, grid, Personne, nb_personnes[-1], debug) # On lit et positionne le nombre de personne souhaité seule personne
    debug.affiche("nbre de personnes", len(grid.personneList))

    start = time.time()
    grid.Simulation()
    end = time.time()
    duration_1 += [end-start]
    print("La simulation a duré", duration_1[-1], "secondes")
    print("Il restait", grid.waitersCount, "personnes immobiles")

# Vous ferez un plot de l'évolution des temps d'exécution
plt.plot(nb_personnes, duration_1, label="Une seule thread")
#plt.plot(nb_personnes, duration_2, label="Une thread par personne")
#plt.plot(nb_personnes, duration_2, label="Une thread par quard de terrain")
plt.xlabel("nombre de personnes")
plt.ylabel("temps en secondes")
plt.legend()
plt.show()


#cell 81
import sys

class Grille_2(Grille):
    waiters = 0
    threads = []
    
    # Crée un objet list
    def __init__(self, Xmax, Ymax, UI, debug):
        Grille.__init__(self, Xmax, Ymax, UI, debug)
        self.waitersLock = threading.Lock();
        self.mapLock = threading.Lock();
        self.useAvailable = threading.Condition(self.mapLock)
        
    def addWaiter(self):
        with self.waitersLock :
            self.waiters+=1
            
            # si c'est la dernière personne encore en mouvement, 
            # arrete la thread courante après avoir
            # reveilé les autre en attente pour qu'ils se terminent à leur tour
            if self.waiters>=len(self.personneList):
                if self.mapLock.locked() :
                    self.useAvailable.notifyAll()
                sys.exit(1)
        
    def removeWaiter(self):
        with self.waitersLock :
            self.waiters-=1
    
    def removePersonne(self, personne):
        with self.waitersLock :
            Grille.removePersonne(self, personne)
        
        
    def isCellEmpty(self, x, y):
        # les personne ne se considère plus comme des obstacles les unes des autres
        return self.Get(x,y)!=const.OBSTACLE
    
    def Get(self, x, y):
        return self.map[x][y]
    
    def Set(self, x, y, status):
        with self.mapLock :
            if status == const.EMPTY:
                # place est libéré
                Grille.Set(self, x, y, status)
                self.useAvailable.notifyAll()
            else :
                if status == const.PERSONNE:
                    # une personne convoite une place
                    while self.Get(x,y) != const.EMPTY:
                        # la place est prise 
                        # je me met en attente et m'enregistre en tant que tel
                        self.addWaiter()
                        self.useAvailable.wait()
                        self.removeWaiter()
                Grille.Set(self, x, y, status)
            
    # lance la simulation de la personne sur une nouvelle thread
    def RunThread(self, personne):
        x = threading.Thread(target=self.PersonneSimulation, args=(personne,))
        self.threads.append(x)
        x.start()
            
    # simulation d'une seul personne
    def PersonneSimulation(self, personne):
        self.debug.affiche("Une personne vient de se mettre a marcher")
        while(True):
            try :
                end, immobile = personne.deplacePersonne(self, self.debug)
            except Exception as e:
                self.debug.affiche(bcolors.BOLD,"ERROR APPEND HERE",bcolors.ENDC)
                raise e

            if immobile:
                self.debug.affiche("Une personne vient de se bloquer") # face à un mur
                self.addWaiter()
                break
            if end:
                self.debug.affiche("Une personne vient de sortir, il reste", len(self.personneList), "personnes")
                break


    def Simulation(self):
        if self.debug.debug or self.UI==affichage.OpenCV:
            self.mapList.append(copy.copy(self.map))
        if self.UI==affichage.OpenCV:
            cv2.imshow("img", createImage(self.map))
            cv2.waitKey(500)

        # run all thread
        personneListCopy = self.personneList.copy()
        for personne in personneListCopy:                
            self.RunThread(personne)
        # wait for all thread
        while len(self.threads)>0: 
            self.threads.pop(0).join()

        if self.debug.debug or self.UI==affichage.OpenCV:
                self.mapList.append(copy.copy(self.map))
                if self.UI==affichage.OpenCV:
                    cv2.imshow("img", createImage(self.map))
                    cv2.waitKey(500)

        if self.UI==affichage.OpenCV:
            cv2.destroyAllWindows()

#cell 82
class Personne_2(Personne):
    def __init__(self, grid, Xinit, Yinit, Xsortie, Ysortie, debug):
        Personne.__init__(self, grid, Xinit, Yinit, Xsortie, Ysortie, debug)

    def Move(self, grid, depl):
        # place la personne sur la nouvelle case AVANT de liberer l'ancienne
        debug.affiche("Déplacement de la personne de (", self.X, self.Y, ") à (", depl[1], depl[2], ")")
        grid.Set(depl[1], depl[2], const.PERSONNE)
        grid.Set(self.X, self.Y, const.EMPTY)
        self.X = depl[1]
        self.Y = depl[2]

#cell 83
# En mode debug avec 1 personne, sans UI
debug = Debug(True)

file = open_file(petite_grille, debug)  # On ouvre le fichier
X, Y = read_grid_size(file, debug) # On lit la taille
grid = Grille_2(X, Y, affichage.Aucun, debug)     # On créée l'objet grille
read_obstacles(file, grid, Obstacle, debug)  # On lit et positionne les obstacles
read_personnes(file, grid, Personne_2, -1, debug) # On lit et positionne une seule personne
print("*"*20)
print("Début de la simulation")
grid.Simulation()
print("Fin de la simulation")
print("Il restait", grid.waiters, "personnes immobiles")

#cell 84
for img in grid.mapList:
    plt.imshow(img, cmap='Paired', vmin=0, vmax=5)
    plt.show()

#cell 85
''' plot du temps d'exécution en fonction du nombre de personnes '''

debug = Debug(False)

duration_2 = []  # i est le numéro du scénario
nb_personnes = []
for i in range(11):
    nb_personnes += [np.power(2, i)]
    if nb_personnes[-1] != 8 and False:
        continue
    print("*"*30)
    print("pour", nb_personnes[-1], "personnes")

    file = open_file(grande_grille, debug)  # On ouvre le fichier
    X, Y = read_grid_size(file, debug) # On lit la taille
    debug.affiche("simulation pour une grille:", X, Y)
    grid = Grille_2(X, Y,affichage.Aucun, debug)     # On créée l'objet grille
    read_obstacles(file, grid, Obstacle, debug)  # On lit et positionne les obstacles
    read_personnes(file, grid, Personne_2, nb_personnes[-1], debug) # On lit et positionne le nombre de personne souhaité seule personne
    debug.affiche("nbre de personnes", len(grid.personneList))

    start = time.time()
    grid.Simulation()
    end = time.time()
    duration_2 += [end-start]
    print("La simulation a duré", duration_2[-1], "secondes")
    print("Il restait", grid.waiters, "personnes immobiles")

#Vous ferez un plot de l'évolution des temps d'exécution
#plt.plot(nb_personnes, duration_1, label="Une seule thread")
plt.plot(nb_personnes, duration_2, label="Une thread par personne")
#plt.plot(nb_personnes, duration_2, label="Une thread par quard de terrain")
plt.xlabel("nombre de personnes")
plt.ylabel("temps en secondes")
plt.legend()
plt.show()

#cell 91
import sys,json
import ipyparams

def convertNotebook(notebookPath, modulePath):
    f = open(notebookPath, 'r') #input.ipynb
    j = json.load(f)

    of = open(modulePath, 'w') #output.py

    if j["nbformat"] >=4:
        for i,cell in enumerate(j["cells"]):
            if cell['cell_type']=='code':
                of.write("#cell "+str(i+1)+"\n")
                for line in cell["source"]:
                    of.write(line)
                of.write('\n\n')
    else:
        print("not implemented")

    of.close()

#cell 92
''' Remarks: it is necessary to put the import code in a separate cell above the call of function.
Otherwise Jupyter will try to run the code before the library is fully loaded,
so the currentNotebook variable is still blank.

Putting the import in its own cell forces it to load before moving to the next cell.
'''
if is_running_from_ipython():
    currentNotebook = ipyparams.notebook_name
    pythonName = currentNotebook.split(".")[0]+".py"
    
    convertNotebook(currentNotebook, pythonName)

#cell 93


