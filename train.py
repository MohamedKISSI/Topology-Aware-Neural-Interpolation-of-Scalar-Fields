from paraview.simple import *
import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import random
import time
import os
import pickle
import subprocess
import gzip

from utils import *
from topological_function import *
from model import * 


def gradient_x(img):
    return img[:, :, :, 1:] - img[:, :, :, :-1]

def gradient_y(img):
    return img[:, :, 1:, :] - img[:, :, :-1, :]

def gradient_loss(pred, target):
    grad_x_pred = gradient_x(pred)
    grad_y_pred = gradient_y(pred)

    grad_x_target = gradient_x(target)
    grad_y_target = gradient_y(target)

    loss_x = F.smooth_l1_loss(grad_x_pred, grad_x_target)
    loss_y = F.smooth_l1_loss(grad_y_pred, grad_y_target)

    return loss_x + loss_y


if __name__ == "__main__":
    
    # Fix seed
    set_seed(42)

    # Get Threads numbers
    threads_number = os.cpu_count()
    print(f"Threads number : {threads_number}")

    # Parameters 
    num_epochs = 6000
    lr = 0.0005
    dataset = "vonKarman"
    dimension = 512
    alpha_mse = 1
    alpha_criticalPoint = 1
    alpha_W2 = 1

    shuffle = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


    expected_shape = (dimension, dimension)
    keyframe = [0, 16, 33, 49, 66, 82, 99, 115, 132, 149]

    if (alpha_criticalPoint != 0) or (alpha_W2 != 0):
        no_keyframe = [i for i in range(150) if i not in keyframe]
    else:
        no_keyframe = []

    # Constraint on the criticals points
    constraintsCriticalPoint = {}

    for k in keyframe:
        constraintsCriticalPoint[k] = computeConstraintsCriticalPoint(k, dataset, dimension)

    for k in no_keyframe:
        constraintsCriticalPoint[k] = computeConstraintsCriticalPoint(k, dataset, dimension)



    # Import data 
    max_key = max(keyframe + no_keyframe) + 1
    print(f"max_time_steps = {max_key} \n")

    number_of_keyframe = len(keyframe) + len(no_keyframe)

    listTensorData = []
    listTensorkey = []
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

        listTensorData.append(tensor_data)
        listTensorkey.append(torch.tensor(int(file_name)))

        dict_data[int(file_name)] = tensor_data

    processed_images_keyframe = [image for image in listTensorData]

    if not isinstance(listTensorkey, torch.Tensor):
        listTensorkey = torch.tensor(listTensorkey)

    dataset_keyframe = TensorDataset(torch.stack(processed_images_keyframe), listTensorkey)
    dataloader = DataLoader(dataset_keyframe, batch_size=number_of_keyframe, shuffle=shuffle, worker_init_fn=lambda x: set_seed(42))
    dataloader_fintuning = DataLoader(dataset_keyframe, batch_size=number_of_keyframe, shuffle=shuffle, worker_init_fn=lambda x: set_seed(42))

    #===============================================
    #   Training phase 1 : MSE & Criticals points
    #===============================================

    start_time = time.time()

    model = TimeToImageGenerator_512(time_embedding_dim=128, max_time_steps=max_key).to(device)

    weight_decay = 1e-6
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    loss_history = []
    loss_history_mse = []
    loss_history_grad = []
    loss_history_cp = []

    for epoch in range(num_epochs):
        start_time_epoch = time.time()

        for batch_idx, (targets, keys) in enumerate(dataloader):

            keys = keys.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            torch.cuda.empty_cache()

            generated_img = model(keys)


            is_keyframes = torch.isin(keys, torch.tensor(keyframe).to(device))
            is_not_keyframes = ~is_keyframes

            targets_keyframe = targets[is_keyframes]
            targets_no_keyframe = targets[is_not_keyframes]

            outputs_keyframe = generated_img[is_keyframes]
            outputs_no_keyframe = generated_img[is_not_keyframes]

            if(len(outputs_keyframe) == 0):
                mse_loss = torch.tensor(0.0, device=outputs_keyframe.device)
            else:
                mse_loss = criterion(outputs_keyframe, targets_keyframe)
                loss_grad = gradient_loss(outputs_keyframe, targets_keyframe)


            loss = alpha_mse * mse_loss + loss_grad


            loss_critical_point = torch.tensor(0.0, device=outputs_keyframe.device)

            if(alpha_criticalPoint != 0):
                results_loss_topologique_critical_point = []
                for l, key in zip(range(len(keys)), keys):
                    results_loss_topologique_critical_point.append(Loss_topologicalLossesConstrainedOnThePositionsCriticalPoints(generated_img[l], constraintsCriticalPoint[key.item()][0], constraintsCriticalPoint[key.item()][1], expected_shape, device))

                loss_critical_point = torch.mean(torch.stack(results_loss_topologique_critical_point))

                loss += alpha_criticalPoint * loss_critical_point

            loss.backward()
            optimizer.step()

            end_time_epoch = time.time()
            execution_time_epoch = end_time_epoch - start_time_epoch
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.6f}, mse: {mse_loss.item():.6f}, loss_critical_point: {loss_critical_point.item():.6f},  loss_grad: {loss_grad.item():.6f}, time : {execution_time_epoch}")

        loss_history.append(loss.item())
        loss_history_mse.append(mse_loss.item())
        loss_history_grad.append(loss_grad.item())
        loss_history_cp.append(loss_critical_point.item())


    model_phase_1_file = f'model_phase_1'

    # Save with compression
    with gzip.open(f"result_folder/{model_phase_1_file}.pth.gz", 'wb') as f:
        pickle.dump(model.state_dict(), f)


    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Phase 1 : Execution time: {execution_time} seconds")


    #===============================================
    #               Training phase 2 : 
    #           MSE & Criticals points & W2
    #===============================================


    model = TimeToImageGenerator_512(time_embedding_dim=128, max_time_steps=max_key).to(device)
    with gzip.open(f"result_folder/{model_phase_1_file}.pth.gz", 'rb') as f:
        model.load_state_dict(pickle.load(f))

    model.train()

    for param in model.time_projection.parameters():
        param.requires_grad = False

    for layer in list(model.cnn_decoder.children())[:6]:
        for param in layer.parameters():
            param.requires_grad = False

    weight_decay = 1e-6
    lr_fintuning=1e-5
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr_fintuning, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    start_time = time.time()

    loss_history = []
    loss_history_mse = []
    loss_history_W2 = []
    loss_history_cp = []

    num_epochs_fintuning = 100

    for epoch in range(num_epochs_fintuning):
        start_time_epoch = time.time()

        for batch_idx, (targets, keys) in enumerate(dataloader_fintuning):

            keys = keys.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            torch.cuda.empty_cache()

            generated_img = model(keys)


            is_keyframes = torch.isin(keys, torch.tensor(keyframe).to(device))
            is_not_keyframes = ~is_keyframes

            targets_keyframe = targets[is_keyframes]
            targets_no_keyframe = targets[is_not_keyframes]

            outputs_keyframe = generated_img[is_keyframes]
            outputs_no_keyframe = generated_img[is_not_keyframes]

            # Loss MSE (only on keyframes)
            if(len(outputs_keyframe) == 0):
                mse_loss = torch.tensor(0.0, device=outputs_keyframe.device)
            else:
                mse_loss = criterion(outputs_keyframe, targets_keyframe)

            loss = alpha_mse * mse_loss

            # Loss criticals points
            loss_critical_point = torch.tensor(0.0, device=outputs_keyframe.device)

            if(alpha_criticalPoint != 0):
                # Loss Crtical point (keyframes and no keyframes)
                results_loss_topologique_critical_point = []
                for l, key in zip(range(len(keys)), keys):
                    results_loss_topologique_critical_point.append(Loss_topologicalLossesConstrainedOnThePositionsCriticalPoints(generated_img[l], constraintsCriticalPoint[key.item()][0], constraintsCriticalPoint[key.item()][1], expected_shape, device))

                loss_critical_point = torch.mean(torch.stack(results_loss_topologique_critical_point))

                loss += alpha_criticalPoint * loss_critical_point

            # Loss W2
            loss_topologique_W2 = torch.tensor(0.0, device=outputs_keyframe.device)

            start_epoch = 0
            if (epoch >= start_epoch) and (alpha_W2 != 0):

                keys_detach = keys.detach().cpu().numpy()
                outputs_detach = generated_img.detach().cpu().numpy()

                spacing_ = [spacing] * len(outputs_detach)
                origin_ = [origin] * len(outputs_detach)
                dimensions_ = [dimensions] * len(outputs_detach)
                dossier = "."
                dossier_ = [dossier] * len(outputs_detach)
                dataset_ = [dataset] * len(outputs_detach)

                processes = []
                results_files = []


            for idx, args in enumerate(zip(outputs_detach, keys_detach, spacing_, origin_, dimensions_, dossier_, dataset_)):
                input_file = f"./data/script/input_{args[1]}.pkl"
                output_file = f"./data/script/output_{args[1]}.pkl"

                with open(input_file, "wb") as f:
                    pickle.dump(args, f)

                process = subprocess.Popen(
                    ["python", "./data/script/compute_constraints_worker_new.py", input_file, output_file],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
                )

                processes.append((process, input_file, output_file))

            results = []
            for process, input_file, output_file in processes:
                process.wait()
                with open(output_file, "rb") as f:
                    results.append(pickle.load(f))

                os.remove(output_file)
                os.remove(input_file)

            results_loss_topologique_W2 = [
                Loss_topologicalLossesConstrainedOnTheDiagram(generated_img[k], result_keyframe[0], result_keyframe[1], expected_shape, device)
                for k, result_keyframe in enumerate(results)
            ]


            loss_topologique_W2 = torch.mean(torch.stack(results_loss_topologique_W2))

            loss += alpha_W2 * loss_topologique_W2

            loss.backward()
            optimizer.step()

            end_time_epoch = time.time()
            execution_time_epoch = end_time_epoch - start_time_epoch
            print(f"Epoch {epoch + 1}/{num_epochs_fintuning}, Loss: {loss.item():.6f}, mse: {mse_loss.item():.6f}, loss_critical_point: {loss_critical_point.item():.6f}, loss_topologique_W2: {loss_topologique_W2.item():.6f}, time : {execution_time_epoch}")

        loss_history_W2.append(loss_topologique_W2.item())
        loss_history.append(loss.item())
        loss_history_mse.append(mse_loss.item())
        loss_history_cp.append(loss_critical_point.item())


    model_phase_2_file = f"model_phase_2"

    with gzip.open(f"result_folder/{model_phase_2_file}.pth.gz", 'wb') as f:
        pickle.dump(model.state_dict(), f)

    end_time = time.time()
    execution_time_phase_2 = end_time - start_time
    print(f"Execution time: {execution_time_phase_2} seconds")

    x_iters = np.arange(len(loss_history))[::10]

    loss_sampled = loss_history[::10]
    plt.figure(figsize=(8, 6))
    plt.plot(x_iters, loss_sampled, marker='o', linestyle='-')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Epochs (log scale)')
    plt.ylabel('Loss (log scale)')
    plt.title('Loss Evolution During Training (Log-Log Scale)')
    plt.grid(True, which="both", linestyle='--')
    plt.legend(['Phase 2: Training Total Loss'])
    plt.savefig(f"result_folder/total_loss_evolution_phase_2.png")
    plt.show()

