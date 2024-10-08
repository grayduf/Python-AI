x = [1.0, -2.0, 3.0] #inputs 
w = [-4.0, -2.5, 1.0] #weights for inputs
b = 1.0 #bias for neuron

#multiplying inputs by weights 
xw0 = x[0] * w[0] 
xw1 = x[1] * w[1]
xw2 = x[2] * w[2]
print(xw0, xw1, xw2, b)

#sum of inputs*weights + bias
z = xw0 + xw1 + xw2 + b
print(z)

#ReLU function activation
y = max(z, 0)
print(y)

#backwards pass
dvalue = 1.0

#derivative of ReLU and the chain rule
drelu_dz = dvalue * (1. if z > 0 else 0.)
print(drelu_dz)

#partial derivatives of the multiplication
dsum_dxw0 = 1
dsum_dxw1 = 1
dsum_dxw2 = 1
dsum_db = 1
drelu_dxw0 = drelu_dz * dsum_dxw0
drelu_dxw1 = drelu_dz * dsum_dxw1
drelu_dxw2 = drelu_dz * dsum_dxw2
drelu_db = drelu_dz * dsum_db
print(drelu_dxw0, drelu_dxw1, drelu_dxw2, drelu_db)

dmul_dx0 = w[0]
dmul_dx1 = w[1]
dmul_dx2 = w[2]
dmul_dw0 = x[0]
dmul_dw1 = x[1]
dmul_dw2 = x[2]

drelu_dx0 = drelu_dxw0 * dmul_dx0
drelu_dw0 = drelu_dxw0 * dmul_dw0
drelu_dx1 = drelu_dxw1 * dmul_dx1
drelu_dw1 = drelu_dxw1 * dmul_dw1
drelu_dx2 = drelu_dxw2 * dmul_dx2
drelu_dw2 = drelu_dxw2 * dmul_dw2
print(drelu_dx0, drelu_dw0, drelu_dx1, drelu_dw1, drelu_dx2, drelu_dw2)

# way to sum up everything all the way in one function
# drelu_dx0 = dvalue * (1. if z > 0 else 0.) * w[0]

dx = [drelu_dx0, drelu_dx1, drelu_dx2] #gradients on inputs
dw = [drelu_dw0, drelu_dw1, drelu_dw2] #gradients on weights
db = drelu_db #gradient on bias (only one since its just one neuron)



