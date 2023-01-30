# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# # Make a sphere
# x, y, z = np.ogrid[-1:1:512j, -1:1:512j, -1:1:304j]
# sphere = np.sqrt(x * x + y * y + z * z) < 0.5
# # Make 3D axis
# front3d_frustum_mask = np.load(str("data/frustum_mask.npz"))["mask"]
# ax = plt.figure().add_subplot(projection='3d')
# # Make voxels figures at 10% resolution
# ax.voxels(filled=front3d_frustum_mask[::, ::, ::])
# ax.figure.show()
img_dir ='/home/sonic/PycharmProjects/front3d/'

file = open("./resources/front3d/test_list_3d.txt",'r')
cnt= 0
while True:
    line = file.readline()
    if not line:
        break
    print(img_dir + line + '.png')
    sp=line.strip()
    sp=sp.split('/')
    print('output/' +str(cnt).zfill(4)+'/')
    #print(line.split('/')[-1])
    cnt+=1


file.close()