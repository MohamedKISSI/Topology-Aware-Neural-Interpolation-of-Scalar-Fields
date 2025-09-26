from paraview.simple import *
import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import random
from multiprocessing import Pool
import multiprocessing
import time
import vtk
from vtk.util.numpy_support import numpy_to_vtk
import gc
import os
import sys
from torch.cuda.amp import autocast, GradScaler
import pickle
from joblib import Parallel, delayed
import subprocess
import torch
import gzip

from utils import *

set_seed(42)

def computeConstraintsCriticalPoint(key, dataset, dimension):
    targetPdFileName = f"./data/{dataset}/diagram/{dimension}/{key}.vtu"

    # create a new 'XML Unstructured Grid Reader'
    targetDiagram = XMLUnstructuredGridReader(registrationName='diagram2.vtu', FileName=[targetPdFileName])
    targetDiagram.CellArrayStatus = ['PairIdentifier', 'PairType', 'Persistence', 'Birth', 'IsFinite']
    targetDiagram.PointArrayStatus = ['ttkVertexScalarField', 'CriticalType', 'Coordinates']

    # Properties modified on diagram2
    targetDiagram.TimeArray = 'None'

    UpdatePipeline(time=0.0, proxy=targetDiagram)

    targetDiagramObject = servermanager.Fetch(targetDiagram)

    numberOfCells = targetDiagramObject.GetNumberOfCells()
    pointDataTargetDiagram = targetDiagramObject.GetPointData()

    constrained_points_critical_point = []
    target_values_critical_point = []

    for i in range(numberOfCells):

        if targetDiagramObject.GetCellData().GetArray('PairType').GetTuple1(i) == -1:
            continue

        cell  = targetDiagramObject.GetCell(i)

        birthIdPairTargetDiagram = cell.GetPointIds().GetId(0)
        deathIdPairTargetDiagram = cell.GetPointIds().GetId(1)


        indexBirthPairTargetDiagram =  int(pointDataTargetDiagram.GetArray('ttkVertexScalarField').GetTuple1(birthIdPairTargetDiagram))
        indexDeathPairTargetDiagram =  int(pointDataTargetDiagram.GetArray('ttkVertexScalarField').GetTuple1(deathIdPairTargetDiagram))

        (xPoint0, yPoint0, zPoint0) = cell.GetPoints().GetPoint(0)
        (xPoint1, yPoint1, zPoint1) = cell.GetPoints().GetPoint(1)

        pairTargetDiagramPersistence = [targetDiagramObject.GetCellData().GetArray('Persistence').GetTuple1(i)]

        targetValueBirth = xPoint1
        targetValueDeath = xPoint1+pairTargetDiagramPersistence[0]

        constrained_points_critical_point.append(indexBirthPairTargetDiagram)
        target_values_critical_point.append(targetValueBirth)
        constrained_points_critical_point.append(indexDeathPairTargetDiagram)
        target_values_critical_point.append(targetValueDeath)

    # Explicitly free large objects to save memory
    Delete(targetDiagram)
    del targetDiagram, targetDiagramObject, cell, pointDataTargetDiagram
    gc.collect()  # Force memory release

    return constrained_points_critical_point, target_values_critical_point


def computeWassersteinDistance(resultFilePath, time, dataset, dimension):
  # create a new 'XML Image Data Reader'
  targetData = XMLImageDataReader(registrationName=f'{time}.vti', FileName=[f"./data/{dataset}/scalar/{dimension}/{time}.vti"])

  # Properties modified on targetData
  targetData.TimeArray = 'None'

  UpdatePipeline(time=0.0, proxy=targetData)

  # create a new 'TTK PersistenceDiagram'
  tTKPersistenceDiagram1 = TTKPersistenceDiagram(registrationName='TTKPersistenceDiagram1', Input=targetData)

  UpdatePipeline(time=0.0, proxy=tTKPersistenceDiagram1)

  # set active source
  SetActiveSource(targetData)

  # create a new 'XML Image Data Reader'
  resultData = XMLImageDataReader(registrationName='resultData.vti', FileName=[resultFilePath])

  # Properties modified on resultData
  resultData.TimeArray = 'None'

  UpdatePipeline(time=0.0, proxy=resultData)

  # create a new 'Calculator'
  calculator1 = Calculator(registrationName='Calculator1', Input=resultData)

  # Properties modified on calculator1
  calculator1.ResultArrayName = 'opti'
  calculator1.Function = 'Scalars_'

  UpdatePipeline(time=0.0, proxy=calculator1)

  # create a new 'TTK TopologicalSimplificationByPersistence'
  tTKTopologicalSimplificationByPersistence1 = TTKTopologicalSimplificationByPersistence(registrationName='TTKTopologicalSimplificationByPersistence1', Input=calculator1)

  # Properties modified on tTKTopologicalSimplificationByPersistence1
  tTKTopologicalSimplificationByPersistence1.PersistenceThreshold = 0.01

  UpdatePipeline(time=0.0, proxy=tTKTopologicalSimplificationByPersistence1)

  # create a new 'TTK PersistenceDiagram'
  tTKPersistenceDiagram2 = TTKPersistenceDiagram(registrationName='TTKPersistenceDiagram2', Input=tTKTopologicalSimplificationByPersistence1)

  UpdatePipeline(time=0.0, proxy=tTKPersistenceDiagram2)

  # set active source
  SetActiveSource(tTKPersistenceDiagram1)

  # create a new 'TTK BlockAggregator'
  tTKBlockAggregator1 = TTKBlockAggregator(registrationName='TTKBlockAggregator1', Input=[tTKPersistenceDiagram2, tTKPersistenceDiagram1])

  UpdatePipeline(time=0.0, proxy=tTKBlockAggregator1)

  # create a new 'TTK PersistenceDiagramClustering'
  tTKPersistenceDiagramClustering1 = TTKPersistenceDiagramClustering(registrationName='TTKPersistenceDiagramClustering1', Input=tTKBlockAggregator1)

  UpdatePipeline(time=0.0, proxy=tTKPersistenceDiagramClustering1)

  # Get the data
  matchings_data = FetchData(OutputPort(tTKPersistenceDiagramClustering1, 2))[0]

  # Get field data
  field_data = matchings_data.GetBlock(0).GetFieldData()

  # Display the Wasserstein distance
  wasserstein_distance = field_data.GetArray("WassersteinDistance").GetValue(0)

  print(f"Time {time} wasserstein_distance : {wasserstein_distance} \n")

  return wasserstein_distance


def Loss_topologicalLossesConstrainedOnThePositionsCriticalPoints(output, constrained_points_critical_point, target_values_critical_point, expected_shape, device):
    i = [(idx//expected_shape[1]) for idx in constrained_points_critical_point]
    j = [idx%expected_shape[1] for idx in constrained_points_critical_point]

    loss = torch.mean((output.squeeze(0).squeeze(0)[i, j] - torch.tensor(target_values_critical_point).to(device)) ** 2)

    return loss


def Loss_topologicalLossesConstrainedOnTheDiagram(output, constrained_points_diagram, target_values_diagram, expected_shape, device):
    i = [(idx//expected_shape[1]) for idx in constrained_points_diagram]
    j = [idx%expected_shape[1] for idx in constrained_points_diagram]

    loss = torch.sqrt(torch.sum((output.squeeze(0).squeeze(0)[i, j] - torch.tensor(target_values_diagram).to(device)) ** 2))

    return loss