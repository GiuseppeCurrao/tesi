from numpy import load
from numpy import asarray
from numpy import expand_dims
from mtcnn.mtcnn import MTCNN
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from PIL import Image
from keras.models import load_model
import cv2
import os



BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

keras = load_model(BASE_DIR + '/facenet_keras.h5')

#queste funzioni permetteranno di estrapolare dai frame del video la posizione della faccia e la creazione degli embedding
def get_embedding(model, face_pixels):
	# scale pixel values
	face_pixels = face_pixels.astype('float32')
	# standardize pixel values across channels (global)
	mean, std = face_pixels.mean(), face_pixels.std()
	face_pixels = (face_pixels - mean) / std
	# transform face into one sample
	samples = expand_dims(face_pixels, axis=0)
	# make prediction to get embedding
	yhat = model.predict(samples)
	return yhat[0]

def extract_face(image, required_size=(160,160)):
    pixels=asarray(image)
    #creiamo il detector lasciando i pesi di default
    detector= MTCNN()
    #passiamo l'immagine al detector
    result= detector.detect_faces(pixels)
    r = []
    if result ==[]:
        return []
    for  i in range(len(result)):
      x1,y1,width, height = result[i]['box']
      x1, y1 = abs(x1), abs(y1)
      x2, y2 = x1+width, y1+height
      faces = (x1, x2, y1, y2)
      face = asarray(faces)
      r.append(face)
    return r

#carichiamo gli embeddings precedentemente creati
data=load(BASE_DIR + '/npz/database_embeddings.npz')
trainX, trainY = data['arr_0'], data['arr_1']
print('Dataset: train=%d' % (trainX.shape[0]))

#normalizziamo i vettori, portando i moduli dei vari valori pari a dimensioni unitarie
# per farlo utilizziamo il normalizzatore offerto da sklearn
in_encoder = Normalizer(norm='l2')
trainX = in_encoder.transform(trainX)

# label encode targets
out_encoder = LabelEncoder()
out_encoder.fit(trainY)
trainY = out_encoder.transform(trainY)

#passiamo ora all'allenamento del modello. Per farlo utilizziamo una linear support vector machine
#per farlo utilizziamo la classe SVC di sklearn
model= SVC(kernel='linear', probability=True)
model.fit(trainX, trainY)

#valutiamo il modello valutando l'errore tra le predizioni effettuate e le classi effettive
yhat_train=model.predict(trainX)
score_train=accuracy_score(trainY, yhat_train)
print('Accuracy: train=%.3f' % (score_train*100))

#passiamo ora al modello un video qualunque e controlliamo se le previsioni sono corrette
cap = cv2.VideoCapture(0)

while(True):
	#Catturiamo l'immagine frame by frame
	ret, frame = cap.read()

	#estrapoliamo la faccia
	faces = extract_face(frame)
 
	for f in faces:
		x1, x2, y1, y2= f[0], f[1], f[2], f[3]
        
		#x1 e y1 rappresentano ascissa e ordinata di partenza, x2 e y2 quelle finali
		pixels=asarray(frame)
		face = pixels[y1:y2, x1:x2]
		image = Image.fromarray(face)
		image = image.resize((160,160))
		roi = asarray(image)

		#applichiamo il riconoscimento
		samples = get_embedding(keras, roi)
		samples = expand_dims(samples, axis = 0)
		id_ = model.predict(samples)
		class_index = id_[0]
		conf = model.predict_proba(samples)
		class_prob = conf[0, class_index]*100
		predict_names = out_encoder.inverse_transform(id_)

		font = cv2.FONT_HERSHEY_SIMPLEX
		stroke = 2
		colorT=(255, 255, 255)
		#se il valore di confindence è maggiore di una certa soglia scriviamo la previsione sull'immagine
		if class_prob >= 90:
			color = (0, 255, 0)
			name = predict_names
		else:
			color = (0, 0, 255)
			name = "Non riconosciuto"
		
		cv2.putText(frame, str(name), (x1, y1-30), font, 1 , colorT, stroke, cv2.LINE_AA)
        

		#disegnamo il rettangolo attorno alla faccia
		cv2.rectangle(frame, (x1,y1), (x2,y2), color, stroke)

	#Mostriamo il frame risultante
	cv2.imshow("frame", frame)
	if cv2.waitKey(20) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()

