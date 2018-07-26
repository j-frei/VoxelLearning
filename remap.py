import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from dense_3D_spatial_transformer import Dense3DSpatialTransformer
'''
matplotlib.interactive(False)

size = 10

grid = np.empty(size*size*3).reshape(size,size,3).astype(np.float32)
for x in range(size):
    for y in range(size):
        grid[x,y,:] = [x,y,z]

manifold = np.asarray([ x*y for x in range(size) for y in range(size)]).reshape(size,size).astype(np.float32)

dx,dy = np.gradient(manifold)
Fx, Fy = -dx,-dy

Fmax = np.max([np.max(np.abs(Fx)),np.max(np.abs(Fy))])
Fx /= Fmax
Fy /= Fmax

#plt.imshow(manifold)
#for i in range(size)[::3]:
#    for j in range(size)[::3]:
#        plt.arrow(i,j,Fx[i,j],Fy[i,j])

#plt.show()
#plt.imshow(dx)
#plt.show()

# TEST
Fx = np.ones_like(Fx)+1
Fy = np.zeros_like(Fy)

vx = grid[:,:,0] + Fx
vy = grid[:,:,1].T + Fy
print(vx)
print(vy)

xmap = np.copy(vx)
ymap = np.copy(vy)

plt.imshow(manifold)
plt.show()
trf = cv2.remap(manifold,xmap,ymap,cv2.INTER_LINEAR)
plt.imshow(trf)
plt.show()
'''

#with sess.as_default():
#    f = tf.convert_to_tensor((np.array(range(10)).astype(np.float32)-5.)/10.)
#    print(f.eval(session=sess))
#    print(tf.distributions.Normal(loc=0.,scale=.4).prob(f).eval(session=sess))
from DiffeomorphicRegistrationNet import create_model
create_model((60,60,60,2))