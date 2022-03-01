from PIL import Image
from numpy import asarray
from mtcnn.mtcnn import MTCNN
from os import listdir
from os.path import isdir
from matplotlib import pyplot
from numpy import savez_compressed

#definiamo una funzione che indivudui una faccia da una fotografia
def extract_face(filename, required_size=(160,160)):
    #carichiamo l'immagine dal file
    image=Image.open(filename)
    #convertiamola in RGB e successivamente in un array
    image=image.convert('RGB')
    pixels=asarray(image)
    #creiamo il detector lasciando i pesi di default
    detector= MTCNN()
    #passiamo l'immagine al detector
    result= detector.detect_faces(pixels)
    if result ==[]:
        return []
    x1,y1,width, height = result[0]['box']
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1+width, y1+height
    #estrapoliamo la faccia
    face=pixels[y1:y2, x1:x2]
    #ritagliamo la daccia dall'immagine e rendiamola di una dimesione a noi consona
    image= Image.fromarray(face)
    image= image.resize(required_size)
    face_array=asarray(image)
    return face_array

#definiamo una funzione per caricare le foto dalle cartelle
def load_faces(directory):
    faces=list()
    for filename in listdir(directory):
        path = directory + filename
        face = extract_face(path)
        if len(face)!=0:
            faces.append(face)
    return faces

#definiamouna funzione per cercare la directory contenente le altre sottocartelle di classe
def load_dataset(directory):
    X, y = list(), list()
    #numeriamo le cartelle una per classe
    for subdir in listdir(directory):
        path = directory + subdir + '/'
        if not isdir(path):
            continue
        faces=load_faces(path)
        labels =[subdir for _ in range(len(faces))]
        #controllo del punto di arrivo
        print('>caricati %d campioni per la classe: %s' % (len(faces), subdir))
        X.extend(faces)
        y.extend(labels)
    return asarray(X), asarray(y)

folder_train= "C:/Users/Utente/Tesi/Foto/turf faces/TD_RGB_A_Set1/"
folder_test= 'C:/Users/Utente/Tesi/Foto/turf faces/TD_RGB_E_Set1/'
Xtraining, Ytraining= load_dataset(folder_train)
Xtest, Ytest = load_dataset(folder_test)
print(Xtraining.shape, Ytraining.shape)
savez_compressed('TD_RGB_A_Set1', Xtraining, Ytraining, Xtest, Ytest)