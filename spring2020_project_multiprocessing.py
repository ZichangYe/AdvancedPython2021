import numpy as np
from mpi4py import MPI
import face_recognition
import os
from PIL import Image
import cv2
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
dest = 0
path = os.getcwd()
ts = time.time()
# set up parameters
if rank == 0:
    # read the image
    image = face_recognition.load_image_file(path + "/data/worlds-largest-selfie.jpg").astype('float64')
    outputData = np.zeros(image.shape, dtype='float64')
    height, width, color = image.shape
    split = np.array_split(image,size,axis = 0)
    split_sizes = []

    for i in range(0,len(split),1):
        split_sizes = np.append(split_sizes, len(split[i]))
    split_sizes_input = split_sizes*width*color
    displacements_input = np.insert(np.cumsum(split_sizes_input),0,0)[0:-1]

    split_sizes_output = split_sizes*width*color
    displacements_output = np.insert(np.cumsum(split_sizes_output),0,0)[0:-1]

    print("Input data split into vectors of sizes %s" %split_sizes_input)
    print("Input data split with displacements of %s" %displacements_input)
else:
    split_sizes_input = None
    displacements_input = None
    split_sizes_output = None
    displacements_output = None
    split = None
    image = None
    outputData = None
    width = None

split = comm.bcast(split, root=0) #Broadcast split array to other cores
split_sizes = comm.bcast(split_sizes_input, root = 0)
displacements = comm.bcast(displacements_input, root = 0)
split_sizes_output = comm.bcast(split_sizes_output, root = 0)
displacements_output = comm.bcast(displacements_output, root = 0)
width = comm.bcast(width, root=0)
output_chunk = np.zeros(np.shape(split[rank])) #Create array to receive subset of data on each core, where rank specifies the core
# print("Rank %d with output_chunk shape %s" %(rank,output_chunk.shape))

comm.Scatterv([image,split_sizes_input, displacements_input,MPI.DOUBLE], output_chunk, root=0)

output = np.zeros([len(output_chunk),width, 3]) #Create output array on each core

for i in range(0,np.shape(output_chunk)[0],1):
    # print(i)
    # print(output_chunk[i].shape)
    output[i,0:width, 0:3] = output_chunk[i]
# print(f"rank {rank} with output {output[0:5]}")
comm.Barrier()
output = output.astype(np.uint8)

# Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
img_rgb = output[:, :, ::-1]

# image = face_recognition.load_image_file(path + f"/{rank}.png")
face_locations = face_recognition.face_locations(img_rgb, model='knn')
print(f"rank {rank} with {len(face_locations)} faces detected!")

print(time.time() - ts)

# if rank == 0:
#     outputData = outputData[0:len(test),:]
#     print("Final data shape %s" %(outputData.shape,))
#     plt.imshow(outputData)
#     plt.colorbar()
#     plt.show()
#     print(outputData)
