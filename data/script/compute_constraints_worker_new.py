import sys
import pickle
import vtk
import gc
from paraview.simple import *
from vtk.util.numpy_support import numpy_to_vtk
import os 
import numpy as np 
from utils import set_seed

set_seed(42)

def computeConstraints(output, key, spacing, origin, dimensions, dossier, dataset):
    

    vtiFileName = f"./intermediate_file/output_{key}.vti"
    pdFileName = f"./intermediate_file/output_diagram_{key}.vtu"

    #======================================
    #             Numpy to VTI
    #======================================

    output_array = output

    if dimensions[2] > 1: 
        output_array = np.transpose(output_array.squeeze(0) , (1, 2, 0))
   
    imageData = vtk.vtkImageData()

    # Définir les dimensions
    imageData.SetDimensions(dimensions[0], dimensions[1], dimensions[2])
    imageData.SetSpacing(spacing[0], spacing[1], spacing[2])
    imageData.SetOrigin(origin[0], origin[1], origin[2])

    # Convertir le tableau numpy en VTK array
    vtk_array = numpy_to_vtk(num_array=output_array.ravel(), deep=True, array_type=vtk.VTK_FLOAT)

    # Ajouter l'array au vtkImageData
    imageData.GetPointData().SetScalars(vtk_array)

    # Créer un writer pour sauvegarder le fichier en format VTI
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(vtiFileName)
    writer.SetInputData(imageData)
    writer.Write()

    # # # Libérer explicitement les objets pour économiser la mémoire
    # del output_array, vtk_array, imageData, writer
    # gc.collect()  # Forcer la collecte des ordures pour libérer la mémoire immédiatement

    #======================================
    #           Persistence Diagram
    #======================================

    # create a new 'XML Image Data Reader'
    testvti = XMLImageDataReader(registrationName='test.vti', FileName=[vtiFileName])
    testvti.PointArrayStatus = ['Scalars_']

    # Properties modified on testvti
    testvti.TimeArray = 'None'

    UpdatePipeline(time=0.0, proxy=testvti)

    # create a new 'TTK PersistenceDiagram'
    tTKPersistenceDiagram1 = TTKPersistenceDiagram(registrationName='TTKPersistenceDiagram1', Input=testvti)
    tTKPersistenceDiagram1.ScalarField = ['POINTS', 'Scalars_']
    tTKPersistenceDiagram1.InputOffsetField = ['POINTS', 'Scalars_']
    # tTKPersistenceDiagram1.DebugLevel = 0

    tTKPersistenceDiagram1.UseAllCores = 0
    tTKPersistenceDiagram1.ThreadNumber = 1
    UpdatePipeline(time=0.0, proxy=tTKPersistenceDiagram1)

    # #=============================================================================================================================
    # #   Threshold all diagrams at a persistence of 10−2 to reduce the size of the diagrams and speed up distance calculations
    # #=============================================================================================================================
    constrained_points_diagram = []
    target_values_diagram = []

    # create a new 'Threshold'
    # currentDiagram = pair > 0.01
    currentDiagram = Threshold(registrationName='Threshold1', Input=tTKPersistenceDiagram1)

    # Properties modified on currentDiagram
    currentDiagram.Scalars = ['CELLS', 'Persistence']
    currentDiagram.UpperThreshold = 0.01
    currentDiagram.ThresholdMethod = 'Above Upper Threshold'

    UpdatePipeline(time=0.0, proxy=currentDiagram)


    # create a new 'Threshold'
    # threshold2 = pair < 0.01
    threshold2 = Threshold(registrationName='Threshold2', Input=tTKPersistenceDiagram1)

    # Properties modified on threshold2
    threshold2.Scalars = ['CELLS', 'Persistence']
    threshold2.LowerThreshold = 0.01
    threshold2.ThresholdMethod = 'Below Lower Threshold'

    UpdatePipeline(time=0.0, proxy=threshold2)

    DiagramObject = servermanager.Fetch(threshold2)

    numberOfCells = DiagramObject.GetNumberOfCells()
    pointDataDiagram = DiagramObject.GetPointData()

    
    # Récupérer les données de persistence
    persistence_array = DiagramObject.GetCellData().GetArray("Persistence")

    for i in range(numberOfCells):
        persistence = DiagramObject.GetCellData().GetArray("Persistence").GetTuple1(i)

        if persistence > 0.01: 
            print(f"Problem persistence pair > 0.01")
        else: 

            cell = DiagramObject.GetCell(i)

            # On récupere les ids des points 0 et 1 de la cellule i
            birth = cell.GetPointIds().GetId(0)
            death = cell.GetPointIds().GetId(1)

            indexBirth =  int(pointDataDiagram.GetArray('ttkVertexScalarField').GetTuple1(birth))
            indexDeath =  int(pointDataDiagram.GetArray('ttkVertexScalarField').GetTuple1(death))

            (xPoint0, yPoint0, zPoint0) = cell.GetPoints().GetPoint(0)
            (xPoint1, yPoint1, zPoint1) = cell.GetPoints().GetPoint(1)

            targetValueBirth = xPoint1
            targetValueDeath = xPoint1 + persistence

            constrained_points_diagram.append(indexBirth)
            target_values_diagram.append((targetValueBirth + targetValueDeath)/2)

            constrained_points_diagram.append(indexDeath)
            target_values_diagram.append((targetValueBirth + targetValueDeath)/2)

    
    #=======================================
    #               Clustering
    #=======================================

    targetPdFileName = f"./data/{dataset}/diagram/{dimensions[0]}/{key.item()}.vtu"

    # create a new 'XML Unstructured Grid Reader'
    targetDiagram = XMLUnstructuredGridReader(registrationName='diagram2.vtu', FileName=[targetPdFileName])
    targetDiagram.CellArrayStatus = ['PairIdentifier', 'PairType', 'Persistence', 'Birth', 'IsFinite']
    targetDiagram.PointArrayStatus = ['ttkVertexScalarField', 'CriticalType', 'Coordinates']

    # Properties modified on diagram2
    targetDiagram.TimeArray = 'None'

    UpdatePipeline(time=0.0, proxy=targetDiagram)

    # create a new 'Threshold'
    # simplifiedTargetDiagram = pair > 0.01
    simplifiedTargetDiagram = Threshold(registrationName='Threshold1', Input=targetDiagram)

    # Properties modified on currentDiagram
    simplifiedTargetDiagram.Scalars = ['CELLS', 'Persistence']
    simplifiedTargetDiagram.UpperThreshold = 0.01
    simplifiedTargetDiagram.ThresholdMethod = 'Above Upper Threshold'

    UpdatePipeline(time=0.0, proxy=simplifiedTargetDiagram)


    targetDiagramObject = servermanager.Fetch(simplifiedTargetDiagram)
    currentDiagramObject = servermanager.Fetch(currentDiagram)

    # set active source
    SetActiveSource(currentDiagram)

    # create a new 'TTK BlockAggregator'
    tTKBlockAggregator1 = TTKBlockAggregator(registrationName='TTKBlockAggregator1', Input=[currentDiagram, simplifiedTargetDiagram])

    UpdatePipeline(time=0.0, proxy=tTKBlockAggregator1)

    # create a new 'TTK PersistenceDiagramClustering'
    tTKPersistenceDiagramClustering1 = TTKPersistenceDiagramClustering(registrationName='TTKPersistenceDiagramClustering1', Input=tTKBlockAggregator1)

    # Properties modified on tTKPersistenceDiagramClustering1
    tTKPersistenceDiagramClustering1.UseAllCores = 0
    tTKPersistenceDiagramClustering1.ThreadNumber = 1

    # Properties modified on tTKPersistenceDiagramClustering1
    tTKPersistenceDiagramClustering1.Algorithm = 'Classical Auction approach (one cluster only, SLOW)'

    UpdatePipeline(time=0.0, proxy=tTKPersistenceDiagramClustering1)


    PersistenceDiagramClusteringMatching = servermanager.Fetch(tTKPersistenceDiagramClustering1, idx=2)

    #======================================================
    #                Traitement du Block 0
    #======================================================

    block0 = PersistenceDiagramClusteringMatching.GetBlock(0)
    pointDataBlock0  = block0.GetPointData()
    NumberOfCells_Block0 = block0.GetNumberOfCells()
    pointDataNewDiagram = currentDiagramObject.GetPointData()

    # Trouver les matching
    matchingCurrentDiagram = {}
    for i in range(NumberOfCells_Block0):

        if currentDiagramObject.GetCellData().GetArray('PairType').GetTuple1(i) == -1:
            continue

        cell_block0 = block0.GetCell(i)

        # Si le point n'est matché à aucun autre point
        if pointDataBlock0.GetArray('PointID').GetTuple1(cell_block0.GetPointIds().GetId(0)) == -1:
            continue

        # Recuperation de l'id du point 0 dans le diagram de barycentre
        idPairBarycentre = int(pointDataBlock0.GetArray('PointID').GetTuple1(cell_block0.GetPointIds().GetId(0)))
        idPairCurrentDiagram = int(pointDataBlock0.GetArray('PointID').GetTuple1(cell_block0.GetPointIds().GetId(1)))

        # On va maintenant chercher les indices des points de paire dans le nouveau diagram
        pairCurrentDiagram = currentDiagramObject.GetCell(idPairCurrentDiagram)
        # On récupere les ids des points 0 et 1 de la cellule j
        birthIdPairCurrentDiagram = pairCurrentDiagram.GetPointIds().GetId(0)
        deathIdPairCurrentDiagram = pairCurrentDiagram.GetPointIds().GetId(1)
        indexBirthPairCurrentDiagram =  int(pointDataNewDiagram.GetArray('ttkVertexScalarField').GetTuple1(birthIdPairCurrentDiagram))
        indexDeathPairCurrentDiagram =  int(pointDataNewDiagram.GetArray('ttkVertexScalarField').GetTuple1(deathIdPairCurrentDiagram))

        criticalPointsPair = [indexBirthPairCurrentDiagram, indexDeathPairCurrentDiagram]
        pairCurrentDiagramPersistence = [currentDiagramObject.GetCellData().GetArray('Persistence').GetTuple1(idPairCurrentDiagram)]

        (xPoint0, yPoint0, zPoint0) = cell_block0.GetPoints().GetPoint(0)
        (xPoint1, yPoint1, zPoint1) = cell_block0.GetPoints().GetPoint(1)

        targetValueBirth = (xPoint1 + (xPoint1+pairCurrentDiagramPersistence[0]))/2
        targetValueDeath = (xPoint1 + (xPoint1+pairCurrentDiagramPersistence[0]))/2
        targetValue = [targetValueBirth, targetValueDeath]

        matchingCurrentDiagram[idPairBarycentre] = [{'criticalPointsPair':criticalPointsPair, 'targetValue': targetValue, 'persistence': pairCurrentDiagramPersistence}]


    #======================================================
    #                Traitement du Block 1
    #======================================================

    block1 = PersistenceDiagramClusteringMatching.GetBlock(1)
    pointDataBlock1  = block1.GetPointData()
    NumberOfCells_Block1 = block1.GetNumberOfCells()
    pointDataTargetDiagram = targetDiagramObject.GetPointData()

    # Trouver les matching
    matchingTargetDiagram = {}
    for i in range(NumberOfCells_Block1):

        if targetDiagramObject.GetCellData().GetArray('PairType').GetTuple1(i) == -1:
            continue

        cell_block1 = block1.GetCell(i)

        # Si le point n'est matché à aucun autre point
        if pointDataBlock1.GetArray('PointID').GetTuple1(cell_block1.GetPointIds().GetId(0)) == -1:
            continue

        # Recuperation de l'id du point 0 dans le diagram de barycentre
        idPairBarycentre = int(pointDataBlock1.GetArray('PointID').GetTuple1(cell_block1.GetPointIds().GetId(0)))
        idPairTargetDiagram = int(pointDataBlock1.GetArray('PointID').GetTuple1(cell_block1.GetPointIds().GetId(1)))


        # On va maintenant chercher les indices des points de paire dans le nouveau diagram
        pairCurrentDiagram = targetDiagramObject.GetCell(idPairTargetDiagram)
        # On récupere les ids des points 0 et 1 de la cellule j
        birthIdPairTargetDiagram = pairCurrentDiagram.GetPointIds().GetId(0)
        deathIdPairTargetDiagram = pairCurrentDiagram.GetPointIds().GetId(1)

        indexBirthPairTargetDiagram =  int(pointDataTargetDiagram.GetArray('ttkVertexScalarField').GetTuple1(birthIdPairTargetDiagram))
        indexDeathPairTargetDiagram =  int(pointDataTargetDiagram.GetArray('ttkVertexScalarField').GetTuple1(deathIdPairTargetDiagram))

        (xPoint0, yPoint0, zPoint0) = cell_block1.GetPoints().GetPoint(0)
        (xPoint1, yPoint1, zPoint1) = cell_block1.GetPoints().GetPoint(1)

        criticalPointsPair = [indexBirthPairTargetDiagram, indexDeathPairTargetDiagram]
        pairTargetDiagramPersistence = [targetDiagramObject.GetCellData().GetArray('Persistence').GetTuple1(idPairTargetDiagram)]

        targetValueBirth = xPoint1
        targetValueDeath = xPoint1+pairTargetDiagramPersistence[0]
        targetValue = [targetValueBirth, targetValueDeath]

        matchingTargetDiagram[idPairBarycentre] = [{'criticalPointsPair':criticalPointsPair, 'targetValue': targetValue, 'persistence': pairTargetDiagramPersistence}]

    pairNumberPredictionDiagram = NumberOfCells_Block0
    pairNumberTargetDiagram = NumberOfCells_Block1

    #=====================================
    #                Matching
    #=====================================

    matching = matchingCurrentDiagram.copy()
    for key, value in matchingTargetDiagram.items():
        if key in matchingCurrentDiagram:
            matching[key].extend(value)
            # print(f'DUO matching : key {key} => {matching[key][0]} - {matching[key][1]}')
        else:
            matching[key] = value
            # print(f'SEUL matching : key {key} => {matching[key][0]}')


    # # destroy testvti
    # Delete(tTKPersistenceDiagramClustering1)
    # del tTKPersistenceDiagramClustering1

    # # destroy testvti
    # Delete(tTKBlockAggregator1)
    # del tTKBlockAggregator1

    # # destroy testvti
    # Delete(currentDiagram)
    # del currentDiagram

    # # destroy testvti
    # Delete(targetDiagram)
    # del targetDiagram

    # # Libération explicite de l'objet
    # del PersistenceDiagramClusteringMatching
    # gc.collect()

    #=====================================
    #             Find Matching
    #=====================================

    for key, value in matching.items():
        if len(value) == 1:
            # for i in range(2):
            constrained_points_diagram.append(value[0]["criticalPointsPair"][0])
            target_values_diagram.append((value[0]["targetValue"][0] + value[0]["targetValue"][1])/2)

            constrained_points_diagram.append(value[0]["criticalPointsPair"][1])
            target_values_diagram.append((value[0]["targetValue"][0] + value[0]["targetValue"][1])/2)

            # print(f'Pair to delete : {value[0]["criticalPointsPair"][0]} & {value[0]["criticalPointsPair"][1]} | persistence : {value[0]["persistence"][0]} \n')
        elif len(value) == 2:
            for i in range(2):
                constrained_points_diagram.append(value[0]["criticalPointsPair"][i])
                target_values_diagram.append(value[1]["targetValue"][i])

    #  # Libérer les objets volumineux et forcer la collecte des ordures
    # del matching
    # gc.collect()

    os.remove(vtiFileName)

    return constrained_points_diagram, target_values_diagram, pairNumberPredictionDiagram, pairNumberTargetDiagram


if __name__ == "__main__":
    # Charger les arguments passés via pickle
    with open(sys.argv[1], "rb") as f:
        args = pickle.load(f)

    # Exécuter la fonction
    result = computeConstraints(*args)
    
    # outputFile = f"{sys.argv[2]}" 

    # Sauvegarder le résultat
    with open(sys.argv[2], "wb") as f:
        pickle.dump(result, f) 
