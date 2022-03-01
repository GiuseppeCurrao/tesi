import cv2
import dlib
import numpy as np
import torch
from torch import Tensor
import glob
from glob import glob
from os.path import basename
from torch import nn
import tensorflow
from tensorflow import keras

#creiamo il detector delle facce presenti in ogni immagine
detector= dlib.get_frontal_face_detector()

#carichiamo il predictor delle features
predictor= dlib.shape_predictor("C:/Users/Utente/Tesi/shape_predictor_68_face_landmarks.dat")

path= 'C:/Users/Utente/Tesi/Foto/aligned_images_DB'


#Creiamo un dataframe partendo dalle nostre immagini
#Carichiamo le classi che abbiamo, che corrispondono per noi alle varie persone
classes = glob(path + '/*')
classes = [basename(c) for c in classes]

#Creiamo un dizionario per mappare le classi su id numerici
class_dict = {c: i for i, c in enumerate(classes)}

image_path=glob('C:/Users/Utente/Tesi/Foto/aligned_images_DB/*/*/*')
image_path=['/'.join(p.split('/')[5:]) for p in image_path]
p=glob('C:/Users/Utente/Tesi/Foto/aligned_images_DB/*/*/*')


def class_from_path(path):
    _, cl,_,_=path.split('\\')
    return class_dict[cl]

#Andiamo a numerare ogni classe, in modo da poter rinvenire più facilmente le classi    
labels = [class_from_path(im) for im in image_path]
#dataset=pd.DataFrame({'path':image_path, 'label':labels})

#Passiamo ora all'estrapolazione delle feature dalle foto        
images = []
for i in p:
    img=cv2.imread(i)
    if img is not None:
        images.append(img)

#il detector richiede immagini in scala di grigio, per cui convertiamole
grays = []
T=[]

for image in images:    
    gray= cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)
    if gray is not None:
        grays.append(gray)

for i in range(0, len(grays)):
    
    faces = detector(grays[i])

    for face in faces:
        x1=face.left()
        x2=face.top()
        x3=face.right()
        x4=face.bottom()
        
        landmarks = predictor(image=grays[i], box=face)
        
        position=np.zeros((68,2))
        for n in range(0,68):
            
            x=landmarks.part(n).x
            y=landmarks.part(n).y
            position[n]=x,y
            
        T.append(position)

#Normalizziamo i dati di ogni predizione, convertendo i dati prima in un numpy array e poi in un tensore
Aux=np.array(T)
T_position= torch.Tensor(Aux)
for i in range(len(T_position)):
    mx=torch.max(T_position[i],0).values
    c=Tensor(68,2)
    c.fill_(8000.)
    X_service=torch.where(T_position>0,T_position,c)
    mn=torch.min(X_service[i],0).values
    if i==0:
        print(mx, '', mn, '\n')
    T_position[i]=T_position[i].subtract(mn).divide(mx.subtract(mn))
    z=torch.zeros(68,2)
    T_position[i]=torch.where(T_position[i]>0,T_position[i],z)

#impostiamo un seed per avere risultati ripetibili
np.random.seed(1234)
torch.random.manual_seed(1234)

#otteniamo una permutazione casuale dei dati
idx = np.random.permutation(len(T_position))

T_position= T_position[idx]
labels = labels[idx]


#Dividiamo i dati in train, validation e test set
train=0.6*len(T_position)
val=0.1*len(T_position)
test=0.3*len(T_position)

X_train=Tensor(T_position[:train])
Y_train= Tensor(labels[:train])
X_val=Tensor(T_position[train:train+val])
Y_val=Tensor(labels[train:train+val])
X_test=Tensor(T_position[train+val:])
Y_test=Tensor(labels[train+val:])

#creiamo il regressore softmax
class SoftMaxRegressor(nn.Module):
    def __init__(self, in_features, out_classes):
        super(SoftMaxRegressor, self).__init__() 
        self.linear = nn.Linear(in_features,out_classes) 
    def forward(self,x):
        scores = self.linear(x)
        return scores


                   