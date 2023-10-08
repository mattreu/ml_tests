import matplotlib.pyplot as plt
import numpy as np

triangle_X = np.array([0,10,5,0])
triangle_Y = np.array([0,1,7,0])
plt.plot(triangle_X,triangle_Y)
plt.title('triangle')
plt.show()

various_cos_X = np.arange(0, 2*np.pi, 0.1)
various_cos1_Y = np.cos(various_cos_X)
various_cos2_Y = np.cos(various_cos_X)**2
various_cos3_Y = np.cos(various_cos_X)**3
plt.plot(various_cos_X, various_cos1_Y, 'ro', various_cos_X, various_cos2_Y, 'b-', various_cos_X, various_cos3_Y, 'g--')
plt.title('various cos')
plt.show()

three_dimensional_x_base = three_dimensional_y_base = np.arange(-2*np.pi, 2*np.pi, 0.1)
three_dimensional_z_base = np.sin(three_dimensional_x_base)*np.sin(three_dimensional_y_base)*np.exp(-three_dimensional_x_base**2-three_dimensional_y_base**2)
three_dimensional_x, three_dimensional_y = np.meshgrid(three_dimensional_x_base, three_dimensional_y_base)
three_dimensional_z = np.sin(three_dimensional_x)*np.sin(three_dimensional_y)*np.exp(-three_dimensional_x**2-three_dimensional_y**2)
levels = np.linspace(three_dimensional_z.min(), three_dimensional_z.max(), 10)
fig = plt.figure()
ax = fig.add_subplot(221, projection="3d")
ax.plot_surface(three_dimensional_x, three_dimensional_y, three_dimensional_z, vmin = three_dimensional_z.min()*2)
ax = fig.add_subplot(222, projection="3d")
three_dimensional_x_f = np.append(0, three_dimensional_x_base.flatten())
three_dimensional_y_f = np.append(0, three_dimensional_y_base.flatten())
three_dimensional_x_f, three_dimensional_y_f = np.meshgrid(three_dimensional_x_f, three_dimensional_y_f)
three_dimensional_z_f = np.sin(three_dimensional_x_f)*np.sin(three_dimensional_y_f)*np.exp(-three_dimensional_x_f**2-three_dimensional_y_f**2)
ax.plot_trisurf(three_dimensional_x_f.flatten(), three_dimensional_y_f.flatten(), three_dimensional_z_f.flatten(), vmin = three_dimensional_z_f.min())
ax = fig.add_subplot(223)
ax.tricontourf(three_dimensional_x_f.flatten(), three_dimensional_y_f.flatten(), three_dimensional_z_f.flatten(), levels=levels)
ax = fig.add_subplot(224, projection="3d")
ax.plot_surface(three_dimensional_x, three_dimensional_y, three_dimensional_z, vmin = three_dimensional_z.min()*2)
ax.tricontour(three_dimensional_x_f.flatten(), three_dimensional_y_f.flatten(), three_dimensional_z_f.flatten(), levels=levels)
fig.suptitle('three dimensional plots')
plt.show()

three_d_curve_t = np.arange(0,10*np.pi,0.1)
three_d_curve_x = np.sin(three_d_curve_t)
three_d_curve_y = np.cos(three_d_curve_t)
three_d_curve_z = three_d_curve_t/10
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot(three_d_curve_x, three_d_curve_y, three_d_curve_z)
fig.suptitle('3d curve')
plt.show()