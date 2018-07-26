import numpy as np

from DataLoader import loadOASISData
from DiffeomorphicRegistrationNet import create_model
from Preprocessing import readNormalizedVolumeByPath
import random

data = loadOASISData()[:20]

train,val = data[:int(len(data)*0.8)],data[int(len(data)*0.8):]

oasis_shape = readNormalizedVolumeByPath(train[0].get('img')).shape
print(oasis_shape)

train_fixed_img = [ readNormalizedVolumeByPath(i.get('img')) for i in train[:50]]
train_moving_img = [ readNormalizedVolumeByPath( train[random.choice(list(range(len(train_fixed_img))))].get('img'))  for i,_ in enumerate(train_fixed_img)]

val_fixed_img = [ readNormalizedVolumeByPath(i.get('img')) for i in val[:50]]
val_moving_img = [ readNormalizedVolumeByPath( train[random.choice(list(range(len(val_fixed_img))))].get('img'))  for i,_ in enumerate(val_fixed_img)]

train_X = np.asarray([ np.stack([f,m],-1).reshape(*oasis_shape,2) for f,m in zip(train_fixed_img,train_moving_img)])
print(train_X.shape)
train_y =  [
    np.asarray([ np.zeros(shape=(*oasis_shape,3)).astype(np.float32) for f,m in zip(train_fixed_img,train_moving_img)]),
    np.asarray([f.reshape(*oasis_shape,1) for f,m in zip(train_fixed_img,train_moving_img)])
            ]

val_X = np.asarray([ np.stack([f,m],-1).reshape(*oasis_shape,2) for f,m in zip(val_fixed_img,val_moving_img)])
val_y =  [
    np.asarray([  np.zeros(shape=(*oasis_shape,3)).astype(np.float32) for f,m in zip(val_fixed_img,val_moving_img)]),
    np.asarray([f.reshape(*oasis_shape,1) for f,m in zip(val_fixed_img,val_moving_img)])
            ]

print(train_y[1].shape)



model = create_model((*oasis_shape,2))
model.fit(x=train_X,y=train_y,epochs=50,batch_size=1)