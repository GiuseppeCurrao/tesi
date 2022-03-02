from numpy import load
from numpy import expand_dims
from numpy import asarray
from numpy import savez_compressed
from keras.models import load_model
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# definiamo una funzione che ci permetta di estrarre delle feature da ogni faccia del dataset e le salvi in un vettore
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

# carichiamo il dataset
data = load(os.path.join(BASE_DIR, 'npz/database.npz'))
trainX, trainy= data['arr_0'], data['arr_1']
print('Loaded: ', trainX.shape, trainy.shape)

# carichiamo il modello di facenet
model = load_model(BASE_DIR + '/facenet_keras.h5')
print('Loaded Model')

# convertiamo ogni file del dataset in un nuovo array di feature
newTrainX = list()
for face_pixels in trainX:
	embedding = get_embedding(model, face_pixels)
	newTrainX.append(embedding)
newTrainX = asarray(newTrainX)
print(newTrainX.shape)

# salviamo il nuovo database
final_directory = os.path.join(BASE_DIR, "npz/")
savez_compressed(final_directory + 'database_embeddings.npz', newTrainX, trainy)