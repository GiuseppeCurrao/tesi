# develop a classifier for the 5 Celebrity Faces Dataset
from numpy import load
from numpy import asarray
from mtcnn.mtcnn import MTCNN
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from PIL import Image
import cv2
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#definiamo la funzione di estrazione della fccia da un'immagine, servirà dopo
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
    return (x1, x2, y1, y2)

#per prima cosa carichiamo il database
data=load(os.path.join(BASE_DIR, 'npz/database_embeddings.npz'))
trainX, trainY, testX, testY = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
print('Dataset: train=%d, test=%d' % (trainX.shape[0], testX.shape[0]))

#normalizziamo i vettori, portando i moduli dei vari valori pari a dimensioni unitarie
# per farlo utilizziamo il normalizzatore offerto da sklearn
in_encoder = Normalizer(norm='l2')
trainX = in_encoder.transform(trainX)
testX = in_encoder.transform(testX)

out_encoder = LabelEncoder()
out_encoder.fit(trainY)
trainy = out_encoder.transform(trainY)
testy = out_encoder.transform(testY)

#passiamo ora all'allenamento del modello. Per farlo utilizziamo una linear support vector machine
#per farlo utilizziamo la classe SVC di sklearn
model= SVC(kernel='linear', probability=True)
model.fit(trainX, trainY)

#valutiamo il modello valutando l'errore tra le predizioni effettuate e le classi effettive
yhat_train=model.predict(trainX)
yhat_test=model.predict(testX)

score_train=accuracy_score(trainY, yhat_train)
score_test=accuracy_score(testY, yhat_test)

print('Accuracy: train=%.3f, test=%.3f' % (score_train*100, score_test*100))

#testiamolo con un video
#utilizziamo opencv per l'apertura dei video e per scrivere sui vari frame
cap = cv2.VideoCapture("C:/Users/Utente/dev/opencv/images/TD_VIDEO//TD_V_1.mp4")

while(True):
	#Catturiamo l'immagine frame by frame
	ret, frame = cap.read()

	#estrapoliamo la faccia
	faces= extract_face(frame)

	for(x1, x2, y1, y2) in faces:
		#x, y, w e h sono le posizioni della faccia riconosciuta: 
		#x1 e y1 rappresentano ascissa e ordinata di partenza, x2 e y2 quelle finali
		roi = frame[y1:y2, x1:x2]

		#applichiamo il riconoscimento
		samples = expand_dims(roi, axis = 0)
		id_ = model.predict(samples)
		conf = model.predict_proba(samples)*100
		
		#se il valore di confindence è maggiore di una certa soglia scriviamo la previsione sull'immagine
		if conf >= 60:
			font = cv2.FONT_HERSHEY_SIMPLEX
			name = out_encoder.inverse_transform(id_)
			color = (255, 255, 255)
			stroke = 2
			cv2.putText(frame, name, (x, y+10), font, 1 , color, stroke, cv2.LINE_AA)

		#disegnamo il rettangoloa ttorno alla faccia
		color = (255, 0, 0) #è in formato BGR
		stroke = 2
		start_point = (x,y)
		widthFinale = x+w
		heightFinale = y+h
		final_point = (widthFinale, heightFinale)
		cv2.rectangle(frame, start_point, final_point, color, stroke)

	#Mostriamo il frame risultante
	cv2.imshow('frame', frame)
	if cv2.waitKey(20) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()