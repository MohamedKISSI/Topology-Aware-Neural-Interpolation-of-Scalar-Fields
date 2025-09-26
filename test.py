from paraview.simple import *
import pyvista as pv
import numpy as np
import torch
import pickle
import torch
import gzip

from utils import *
from topological_function import *
from model import * 

import pandas as pd

set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


dataset = "vonKarman"
dimension = 512
expected_shape = (512, 512)

keyframe = [0, 16, 33, 49, 66, 82, 99, 115, 132, 149]
no_keyframe = [i for i in range(150) if i not in keyframe]

max_key = max(keyframe + no_keyframe) + 1
print(f"max_time_steps = {max_key} \n")

number_of_keyframe = len(keyframe) + len(no_keyframe)

dict_data = {}
dict_numpy_data = {}

files = []
files_names = []

for k in keyframe:
    files.append(f"./data/{dataset}/scalar/{dimension}/{k}.vti")
    files_names.append(k)

for k in no_keyframe:
    files.append(f"./data/{dataset}/scalar/{dimension}/{k}.vti")
    files_names.append(k)


for file, file_name in zip(files, files_names):
    grid = pv.read(file)

    spacing = grid.GetSpacing()
    origin = grid.GetOrigin()
    dimensions = grid.GetDimensions()

    scalarData = grid.get_array(f'Result')

    numpy_data = np.array(scalarData)

    if numpy_data.size == np.prod(expected_shape):
        numpy_data = numpy_data.reshape(expected_shape)
    else:
        raise ValueError(f"Expected dimensions {expected_shape} do not match data size {numpy_data.size}")

    dict_numpy_data[int(file_name)] = numpy_data

    tensor_data = torch.from_numpy(numpy_data).float()

    tensor_data = tensor_data.unsqueeze(0)

    dict_data[int(file_name)] = tensor_data


    
input_data = torch.tensor(keyframe).to(device)
input_data_no_keyframe = torch.tensor(no_keyframe).to(device)

max_time_steps =  max(keyframe + no_keyframe) + 1

model = TimeToImageGenerator_512(time_embedding_dim=128, max_time_steps=max_time_steps).to(device)

with gzip.open(f"result_folder/model_phase_2.pth.gz", 'rb') as f:
    model.load_state_dict(pickle.load(f))

model.eval()


with torch.no_grad():
  output = model(input_data)

  avg_l2_distance = 0
  avg_psnr_distance = 0
  avg_wasserstein_distance = 0

  for o, k in zip(output, input_data):

    output_image = o.squeeze(0).squeeze(0)

    # Compute L2 distance
    l2_distance = torch.norm(o.cpu() - dict_data[k.item()], p=2)
    mse = torch.mean((o.cpu() - dict_data[k.item()]) ** 2)
    psnr = 10 * torch.log10(1 / mse)

    avg_l2_distance +=  l2_distance
    avg_psnr_distance += psnr

    # print(f"keyframe {k} : L2 = {l2_distance}, PSNR = {psnr}")


    # Save generated keyframe
    resultFilePath = f"result_folder/{k}.vti"
    tensorToVTI(output_image, spacing, origin, dimensions, name=resultFilePath)

    # Compute Wasserstein distance
    wassersteinDistance = computeWassersteinDistance(resultFilePath, k, dataset, dimension)
    avg_wasserstein_distance += wassersteinDistance
    # print(f"Wasserstein distance : {wassersteinDistance} \n")

  avg_l2_distance_phase_2_keyframes = avg_l2_distance / len(output)
  avg_psnr_distance_phase_2_keyframes =  avg_psnr_distance / len(output)
  avg_wasserstein_distance_phase_2_keyframes = avg_wasserstein_distance / len(output)
  print(f"Average L2 distance across keyframes = {avg_l2_distance_phase_2_keyframes}")
  print(f"Average PSNR distance across keyframes = {avg_psnr_distance_phase_2_keyframes}")
  print(f"Average Wasserstein distance across keyframe = {avg_wasserstein_distance_phase_2_keyframes}")

with torch.no_grad():

  output = model(input_data_no_keyframe)

  avg_l2_distance = 0
  avg_psnr_distance = 0
  avg_wasserstein_distance = 0

  for o, k in zip(output, input_data_no_keyframe):

    output_image = o.squeeze(0).squeeze(0)

    # Compute L2 distance
    l2_distance = torch.norm(o.cpu() - dict_data[k.item()], p=2)
    mse = torch.mean((o.cpu() - dict_data[k.item()]) ** 2)
    psnr = 10 * torch.log10(1 / mse)

    avg_l2_distance +=  l2_distance
    avg_psnr_distance += psnr
    # print(f"No keyframe {k} : L2 = {l2_distance}, PSNR = {psnr}")

    # Save generated keyframe
    resultFilePath = f"result_folder/{k}.vti"
    tensorToVTI(output_image, spacing, origin, dimensions, name=resultFilePath)

    # Compute Wasserstein distance
    wassersteinDistance = computeWassersteinDistance(resultFilePath, k, dataset, dimension)
    avg_wasserstein_distance += wassersteinDistance
    # print(f"Wasserstein distance : {wassersteinDistance} \n")

  if len(output) > 0:
    avg_l2_distance_phase_2_no_keyframes = avg_l2_distance / len(output)
    avg_psnr_distance_phase_2_no_keyframes = avg_psnr_distance / len(output)
    avg_wasserstein_distance_phase_2_no_keyframes = avg_wasserstein_distance / len(output)
    print(f"Average L2 distance across no keyframes = {avg_l2_distance_phase_2_no_keyframes}")
    print(f"Average PSNR distance across no keyframes = {avg_psnr_distance_phase_2_no_keyframes}")
    print(f"Average Wasserstein distance across no keyframe = {avg_wasserstein_distance_phase_2_no_keyframes}")



#=====================================
#             Interpolation
#=====================================

list_l2_distance_interpolation = []
list_psnr_distance_interpolation = []
list_W2_distance_interpolation = []

for i in range(len(keyframe)-1):
    before = dict_numpy_data[keyframe[i]]
    after = dict_numpy_data[keyframe[i+1]]

    for no_key in no_keyframe:
        if (no_key > keyframe[i]) and (no_key < keyframe[i+1]):
            interpolation = interpolation_lineaire(before, after, keyframe[i], no_key, keyframe[i+1])

            # L2 distance
            target = dict_numpy_data[no_key]
            l2_distance = np.sqrt(np.sum((interpolation - target) ** 2))
            mse = np.mean((interpolation - target) ** 2)
            psnr = 10 * np.log10(1 / mse)
            print(f"[{keyframe[i]} - {keyframe[i+1]}] no_keyframe = {no_key} : PSNR = {psnr}")

            list_l2_distance_interpolation.append(l2_distance)
            list_psnr_distance_interpolation.append(psnr)
            print(f"[{keyframe[i]} - {keyframe[i+1]}] no_keyframe = {no_key} : L2 = {l2_distance}")

            # Save interpolation
            resultFilePath = f"result_folder_interpolation/{no_key}.vti"
            numpyToVTI(interpolation, spacing, origin, dimensions, name=resultFilePath)

            # Compute Wasserstein distance
            wassersteinDistance = computeWassersteinDistance(resultFilePath, no_key, dataset, dimension)
            list_W2_distance_interpolation.append(wassersteinDistance)

            print(f"Wasserstein distance : {wassersteinDistance} \n")


avg_l2_distance_interpolation = np.mean(list_l2_distance_interpolation)
avg_psnr_distance_interpolation = np.mean(list_psnr_distance_interpolation)
avg_W2_distance_interpolation = np.mean(list_W2_distance_interpolation)

print(f"Interpolation: Average L2 distance on no-keyframes= {avg_l2_distance_interpolation}")
print(f"Interpolation: Average PSNR distance on no-keyframes= {avg_psnr_distance_interpolation}")
print(f"Interpolation: Average Wasserstein distance on no-keyframes = {avg_W2_distance_interpolation}")



data = {
    "L2": [avg_l2_distance_interpolation, avg_l2_distance_phase_2_no_keyframes.item()],
    "PSNR": [avg_psnr_distance_interpolation, avg_psnr_distance_phase_2_no_keyframes.item()],
    "W2": [avg_W2_distance_interpolation, avg_wasserstein_distance_phase_2_no_keyframes]
}

# Créer un DataFrame à partir du dictionnaire
df = pd.DataFrame(data, index=["Interpolation", "Our model"])

# Sauvegarder le DataFrame en CSV
df.to_csv("resultats.csv")
