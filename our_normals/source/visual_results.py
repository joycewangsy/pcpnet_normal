import numpy as np
import os
import torch
txt_path='/Users/apple/Desktop/best'
results_file='/Users/apple/Downloads/pclouds'
point_cloud_file='/Users/apple/Downloads/pclouds'
save_file_path='/Users/apple/Desktop/best/1'
shape_list_filename='testset.txt'

shape_names = []
with open(os.path.join(txt_path, shape_list_filename)) as f:
   shape_names = f.readlines()
   shape_names = [x.strip() for x in shape_names]
   shape_names = list(filter(None, shape_names))

for i in range(len(shape_names)):
    shape_name=shape_names[i]
    point_filename=os.path.join(point_cloud_file, shape_name + '.xyz')
    point_cloud=np.loadtxt(point_filename).astype('float32')
    #ids_filename=os.path.join(results_file, shape_name + '.idx')
    #idx=np.loadtxt(ids_filename).astype('int')
    #idx=torch.from_numpy(idx)
    results_filename=os.path.join(point_cloud_file, shape_name + '.normals')
    normals=np.loadtxt(results_filename).astype('float32')
    #point_cloud_save=point_cloud[idx,:]
    points_files=np.concatenate([point_cloud,normals],axis=1)
    np.savetxt(os.path.join(save_file_path, shape_name + '.xyz'),points_files)

