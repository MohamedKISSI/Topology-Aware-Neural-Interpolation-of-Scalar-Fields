from paraview.simple import *
import vtk
from vtk.util.numpy_support import numpy_to_vtk
import gc
import torch
import numpy as np
import random

#==============================
#           Fix seed 
#==============================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False




def tensorToVTI(output_array, spacing, origin, dimensions, name="output.vti"):
    imageData = vtk.vtkImageData()

    imageData.SetDimensions(dimensions[0], dimensions[1], dimensions[2])
    imageData.SetSpacing(spacing[0], spacing[1], spacing[2])
    imageData.SetOrigin(origin[0], origin[1], origin[2])


    vtk_array = numpy_to_vtk(num_array=output_array.cpu().ravel(), deep=True, array_type=vtk.VTK_FLOAT)

    imageData.GetPointData().SetScalars(vtk_array)

    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(name)
    writer.SetInputData(imageData)
    writer.Write()

    del output_array, vtk_array, imageData, writer
    gc.collect()


def numpyToVTI(output_array, spacing, origin, dimensions, name="output.vti"):
    imageData = vtk.vtkImageData()

    imageData.SetDimensions(dimensions[0], dimensions[1], dimensions[2])
    imageData.SetSpacing(spacing[0], spacing[1], spacing[2])
    imageData.SetOrigin(origin[0], origin[1], origin[2])


    vtk_array = numpy_to_vtk(num_array=output_array.ravel(), deep=True, array_type=vtk.VTK_FLOAT)

    imageData.GetPointData().SetScalars(vtk_array)

    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(name)
    writer.SetInputData(imageData)
    writer.Write()

    del output_array, vtk_array, imageData, writer
    gc.collect()


def computeWassersteinDistance(resultFilePath, time):
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

def interpolation_lineaire(image_t0, image_t2, t0, t1, t2):

    # alpha: Interpolation factor between t0 and t2.
    alpha = (t1-t0)/(t2-t0)

    image_t1 = alpha * image_t2 + (1 - alpha) * image_t0
    return image_t1