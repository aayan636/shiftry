# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import argparse
import datetime
from distutils.dir_util import copy_tree
import os
import shutil
import sys
import operator
import tempfile
import traceback

from seedot.compiler.converter.converter import Converter

import seedot.config as config
from seedot.compiler.compiler import Compiler
from seedot.predictor import Predictor
import seedot.util as Util


class Main:

    def __init__(self, algo, version, target, trainingFile, testingFile, modelDir, sf, maximisingMetric, dataset):
        self.algo, self.version, self.target = algo, version, target
        self.trainingFile, self.testingFile, self.modelDir = trainingFile, testingFile, modelDir
        self.sf = sf
        self.dataset = dataset
        self.accuracy = {}
        self.maximisingMetric = maximisingMetric
        self.variableSubstitutions = {} #evaluated during profiling code run
        self.scalesForX = {} #populated for multiple code generation
        self.variableToBitwidthMap = {} #Populated during profiling code run
        self.sparseMatrixSizes = {} #Populated during profiling code run
        self.varDemoteDetails = [] #Populated during variable demotion in VBW mode
        self.flAccuracy = -1 #Populated during profiling code run
        self.allScales = {} #Eventually populated with scale assignments in final code
        self.demotedVarsList = [] #Populated in VBW mode after exploration completed
        self.demotedVarsOffsets = {} #Populated in VBW mode after exploration completed

    def setup(self):
        curr_dir = os.path.dirname(os.path.realpath(__file__))
        
        copy_tree(os.path.join(curr_dir, "Predictor"), os.path.join(config.tempdir, "Predictor"))

        for fileName in ["arduino.ino", "config.h", "predict.h"]:
            srcFile = os.path.join(curr_dir, "arduino", fileName)
            destFile = os.path.join(config.outdir, fileName)
            shutil.copyfile(srcFile, destFile)

    # Generate the fixed-point code using the input generated from the
    # Converter project
    def compile(self, version, target, sf, generateAllFiles=True, id=None, printSwitch=-1, scaleForX=None, variableToBitwidthMap=None, demotedVarsList=[], demotedVarsOffsets={}):
        print("Generating code...", end='')

        if variableToBitwidthMap is None:
            variableToBitwidthMap = dict(self.variableToBitwidthMap)

        # Set input and output files
        inputFile = os.path.join(self.modelDir, "input.sd")
        profileLogFile = os.path.join(
            config.tempdir, "Predictor", "output", "float", "profile.txt")

        logDir = os.path.join(config.outdir, "output")
        os.makedirs(logDir, exist_ok=True)
        if version == config.Version.floatt:
            outputLogFile = os.path.join(logDir, "log-float.txt")
        else:
            if config.ddsEnabled:
                outputLogFile = os.path.join(logDir, "log-fixed-" + str(abs(scaleForX)) + ".txt")
            else:
                outputLogFile = os.path.join(logDir, "log-fixed-" + str(abs(sf)) + ".txt")

        if target == config.Target.arduino:
            outdir = os.path.join(config.outdir, str(config.wordLength) if version == config.Version.fixed else "float", self.algo, self.dataset)
            os.makedirs(outdir, exist_ok=True)
            outputDir = os.path.join(outdir)
        elif target == config.Target.x86:
            outputDir = os.path.join(config.tempdir, "Predictor")

        try:
            obj = Compiler(self.algo, version, target, inputFile, outputDir,
                           profileLogFile, sf, outputLogFile, 
                           generateAllFiles, id, printSwitch, self.variableSubstitutions, 
                           scaleForX,
                           variableToBitwidthMap, self.sparseMatrixSizes, demotedVarsList, demotedVarsOffsets)
            obj.run()
            self.allScales = dict(obj.varScales)
            if version == config.Version.floatt:
                self.variableSubstitutions = obj.substitutions
                self.variableToBitwidthMap = dict.fromkeys(obj.independentVars, config.wordLength)
        except:
            print("failed!\n")
            traceback.print_exc()
            return False

        if id is None:
            self.scaleForX = obj.scaleForX
        else:
            self.scalesForX[id] = obj.scaleForX

        print("completed")
        return True

    # Run the converter project to generate the input files using reading the
    # training model
    def convert(self, version, datasetType, target, varsForBitwidth={}, demotedVarsOffsets={}):
        print("Generating input files for %s %s dataset..." %
              (version, datasetType), end='')

        # Create output dirs
        if target == config.Target.arduino:
            outputDir = os.path.join(config.outdir, "input")
            datasetOutputDir = outputDir
        elif target == config.Target.x86:
            outputDir = os.path.join(config.tempdir, "Predictor")
            datasetOutputDir = os.path.join(config.tempdir, "Predictor", "input")
        else:
            assert False

        os.makedirs(datasetOutputDir, exist_ok=True)
        os.makedirs(outputDir, exist_ok=True)

        inputFile = os.path.join(self.modelDir, "input.sd")

        try:
            varsForBitwidth = dict(varsForBitwidth)
            for var in demotedVarsOffsets:
                varsForBitwidth[var] = config.wordLength // 2
            obj = Converter(self.algo, version, datasetType, target,
                            datasetOutputDir, outputDir, varsForBitwidth, self.allScales)
            obj.setInput(inputFile, self.modelDir,
                         self.trainingFile, self.testingFile)
            obj.run()
            if version == config.Version.floatt:
                self.sparseMatrixSizes = obj.sparseMatrixSizes
        except Exception as e:
            traceback.print_exc()
            return False

        print("done\n")
        return True

    # Build and run the Predictor project
    def predict(self, version, datasetType):
        outputDir = os.path.join("output", version)

        curDir = os.getcwd()
        os.chdir(os.path.join(config.tempdir, "Predictor"))

        obj = Predictor(self.algo, version, datasetType,
                        outputDir, self.scaleForX, self.scalesForX)
        execMap = obj.run()

        os.chdir(curDir)

        return execMap

    # Compile and run the generated code once for a given scaling factor
    def partialCompile(self, version, target, scale, generateAllFiles, id, printSwitch, variableToBitwidthMap=None, demotedVarsList=[], demotedVarsOffsets={}):
        if config.ddsEnabled:
            res = self.compile(version, target, None, generateAllFiles, id, printSwitch, scale, variableToBitwidthMap, demotedVarsList, demotedVarsOffsets)
        else:
            res = self.compile(version, target, scale, generateAllFiles, id, printSwitch, None, variableToBitwidthMap, demotedVarsList, demotedVarsOffsets)
        if res == False:
            return False
        else:
            return True

    def runAll(self, version, datasetType, codeIdToScaleFactorMap, demotedVarsToOffsetToCodeId=None, doNotSort=False):
        execMap = self.predict(version, datasetType)
        if execMap == None:
            return False, True

        if codeIdToScaleFactorMap is not None:
            for codeId, sf in codeIdToScaleFactorMap.items():
                self.accuracy[sf] = execMap[str(codeId)]
                print("Accuracy at scale factor %d is %.3f%%, Disagreement Count is %d, Reduced Disagreement Count is %d\n" % (sf, execMap[str(codeId)][0], execMap[str(codeId)][1], execMap[str(codeId)][2]))
                if datasetType == config.DatasetType.testing and self.target == config.Target.arduino:
                    outdir = os.path.join(config.outdir, str(config.wordLength), self.algo, self.dataset)
                    os.makedirs(outdir, exist_ok=True)
                    file = open(os.path.join(outdir, "res"), "w")
                    file.write("Demoted Vars:\n")
                    file.write(str(self.demotedVarsOffsets) if hasattr(self, 'demotedVarsOffsets') else "")
                    file.write("\nAll scales:\n")
                    file.write(str(self.allScales))
                    file.write("\nAccuracy at scale factor %d is %.3f%%, Disagreement Count is %d, Reduced Disagreement Count is %d\n" % (sf, execMap[str(codeId)][0], execMap[str(codeId)][1], execMap[str(codeId)][2]))
                    file.close()
        else:
            def getMaximisingMetricValue(a):
                if self.maximisingMetric == config.MaximisingMetric.accuracy:
                    return (a[1][0], -a[1][1], -a[1][2])
                elif self.maximisingMetric == config.MaximisingMetric.disagreements:
                    return (-a[1][1], -a[1][2], a[1][0])
                elif self.maximisingMetric == config.MaximisingMetric.reducedDisagreements:
                    return (-a[1][2], -a[1][1], a[1][0])
            allVars = []
            for demotedVars in demotedVarsToOffsetToCodeId:
                offsetToCodeId = demotedVarsToOffsetToCodeId[demotedVars]
                print("Demoted vars: %s\n" % str(demotedVars))
                
                x = [(i, execMap[str(offsetToCodeId[i])]) for i in offsetToCodeId]
                x.sort(key=getMaximisingMetricValue, reverse=True)
                allVars.append(((demotedVars, x[0][0]), x[0][1]))

                for offset in offsetToCodeId:
                    codeId = offsetToCodeId[offset]
                    print("Offset %d (Code ID %d): Accuracy %.3f%%, Disagreement Count %d, Reduced Disagreement Count %d\n" %(offset, codeId, execMap[str(codeId)][0], execMap[str(codeId)][1], execMap[str(codeId)][2]))
            if not doNotSort:
                allVars.sort(key=getMaximisingMetricValue, reverse=True)
            self.varDemoteDetails = allVars
        return True, False

    # Iterate over multiple scaling factors and store their accuracies
    def performSearch(self):
        start, end = config.maxScaleRange

        lastStageAcc = -1

        fixedPointCounter = 0
        while True:
            fixedPointCounter += 1
            if config.fixedPointVbwIteration:
                print("Will compile until conversion to fixed point. Iteration %d"%fixedPointCounter)
            highestValidScale = start
            firstCompileSuccess = False
            while firstCompileSuccess == False:
                if highestValidScale == end:
                    print("Compilation not possible for any Scale Factor. Abort")
                    return False
                try:
                    firstCompileSuccess = self.partialCompile(config.Version.fixed, config.Target.x86, highestValidScale, True, None, 0, dict(self.variableToBitwidthMap), list(self.demotedVarsList), dict(self.demotedVarsOffsets))
                except:
                    firstCompileSuccess = False
                if firstCompileSuccess:
                    break
                highestValidScale -= 1
                
            lowestValidScale = end + 1
            firstCompileSuccess = False
            while firstCompileSuccess == False:
                try:
                    firstCompileSuccess = self.partialCompile(config.Version.fixed, config.Target.x86, lowestValidScale, True, None, 0, dict(self.variableToBitwidthMap), list(self.demotedVarsList), dict(self.demotedVarsOffsets))
                except:
                    firstCompileSuccess = False
                if firstCompileSuccess:
                    break
                lowestValidScale += 1
                
            #Ignored
            self.partialCompile(config.Version.fixed, config.Target.x86, lowestValidScale, True, None, -1, dict(self.variableToBitwidthMap), list(self.demotedVarsList), dict(self.demotedVarsOffsets))

            # The iterator logic is as follows:
            # Search begins when the first valid scaling factor is found (runOnce returns True)
            # Search ends when the execution fails on a particular scaling factor (runOnce returns False)
            # This is the window where valid scaling factors exist and we
            # select the one with the best accuracy
            numCodes = highestValidScale - lowestValidScale + 1
            codeId = 0
            codeIdToScaleFactorMap = {}
            for i in range(highestValidScale, lowestValidScale - 1, -1):
                if config.ddsEnabled:
                    print("Testing with DDS and scale of X as " + str(i))
                else:
                    print("Testing with max scale factor of " + str(i))

                codeId += 1
                try:
                    compiled = self.partialCompile(
                        config.Version.fixed, config.Target.x86, i, False, codeId, -1 if codeId != numCodes else codeId, dict(self.variableToBitwidthMap), list(self.demotedVarsList), dict(self.demotedVarsOffsets))
                except: #If some code in the middle fails to compile
                    codeId -=1
                    continue
                if compiled == False:
                    return False
                codeIdToScaleFactorMap[codeId] = i

            res, exit = self.runAll(config.Version.fixed, config.DatasetType.training, codeIdToScaleFactorMap)

            if exit == True or res == False:
                return False

            print("\nSearch completed\n")
            print("----------------------------------------------")
            print("Best performing scaling factors with accuracy, disagreement, reduced disagreement:")

            self.sf = self.getBestScale()
            if self.accuracy[self.sf][0] != lastStageAcc:
                lastStageAcc = self.accuracy[self.sf][0]
            elif config.fixedPointVbwIteration:
                print("No difference in iteration %d Stage 2 and iteration %d Stage 1. Stopping search"%(fixedPointCounter-1, fixedPointCounter))
                break 

            if config.vbwEnabled:
                assert config.ddsEnabled, "Currently VBW on maxscale not supported"
                if config.wordLength != 16:
                    assert False, "VBW mode only supported if native bitwidth is 16"
                print("Scales computed in native bitwidth. Starting exploration over other bitwidths.")

                attemptToDemote = [var for var in self.variableToBitwidthMap if (var[-3:] != "val" and var not in self.demotedVarsList)]
                numCodes = 3 * len(attemptToDemote) + (6 if 'X' in attemptToDemote else 0) # 9 offsets tried for X while 3 tried for other variables
                
                self.partialCompile(config.Version.fixed, config.Target.x86, self.sf, True, None, -1 if len(attemptToDemote) > 0 else 0, dict(self.variableToBitwidthMap), list(self.demotedVarsList), dict(self.demotedVarsOffsets))
                codeId = 0
                contentToCodeIdMap = {}
                for demoteVar in attemptToDemote:
                    newbitwidths = dict(self.variableToBitwidthMap)
                    newbitwidths[demoteVar] = config.wordLength // 2
                    if demoteVar + "val" in newbitwidths:
                        newbitwidths[demoteVar + "val"] = config.wordLength // 2
                    for alreadyDemotedVars in self.demotedVarsList: # In subsequent iterations during fixed point compilation, this variable will have the variables demoted during the previous runs
                        newbitwidths[alreadyDemotedVars] = config.wordLength // 2
                    demotedVarsList = [i for i in newbitwidths.keys() if newbitwidths[i] != config.wordLength]
                    demotedVarsOffsets = {}
                    for key in self.demotedVarsList:
                        demotedVarsOffsets[key] = self.demotedVarsOffsets[key]

                    contentToCodeIdMap[tuple(demotedVarsList)] = {}
                    for demOffset in ([0, -1, -2] if demoteVar != 'X' else [0, -1, -2, -3, -4, -5, -6, -7, -8]):
                        codeId += 1
                        for k in demotedVarsList:
                            if k not in self.demotedVarsList:
                                demotedVarsOffsets[k] = demOffset
                        contentToCodeIdMap[tuple(demotedVarsList)][demOffset] = codeId
                        compiled = self.partialCompile(config.Version.fixed, config.Target.x86, self.sf, False, codeId, -1 if codeId != numCodes else codeId, dict(newbitwidths), list(demotedVarsList), dict(demotedVarsOffsets))
                        if compiled == False:
                            print("Variable Bitwidth exploration resulted in a compilation error")
                            return False
                
                res, exit = self.runAll(config.Version.fixed, config.DatasetType.training, None, contentToCodeIdMap)

                self.partialCompile(config.Version.fixed, config.Target.x86, self.sf, True, None, -1 if len(attemptToDemote) > 0 else 0, dict(self.variableToBitwidthMap), list(self.demotedVarsList), dict(self.demotedVarsOffsets))
                contentToCodeIdMap = {}
                demotedVarsOffsets = dict(self.demotedVarsOffsets)
                demotedVarsList = list(self.demotedVarsList)
                codeId = 0
                numCodes = len(attemptToDemote)
                demotedVarsListToOffsets = {}
                for ((demoteVars, offset), metrics) in self.varDemoteDetails:
                    newbitwidths = dict(self.variableToBitwidthMap)   
                    for var in demoteVars:
                        if var not in self.demotedVarsList:
                            newbitwidths[var] = config.wordLength // 2
                            demotedVarsOffsets[var] = offset
                        if var not in demotedVarsList:
                            demotedVarsList.append(var)
                    codeId += 1
                    contentToCodeIdMap[tuple(demotedVarsList)] = {}
                    contentToCodeIdMap[tuple(demotedVarsList)][offset] = codeId
                    demotedVarsListToOffsets[tuple(demotedVarsList)] = dict(demotedVarsOffsets)
                    compiled = self.partialCompile(config.Version.fixed, config.Target.x86, self.sf, False, codeId, -1 if codeId != numCodes else codeId, dict(newbitwidths), list(demotedVarsList), dict(demotedVarsOffsets))
                    if compiled == False:
                        print("Variable Bitwidth exploration resulted in another compilation error")
                        return False

                res, exit = self.runAll(config.Version.fixed, config.DatasetType.training, None, contentToCodeIdMap, True)

                if exit == True or res == False:
                    return False

                okToDemote = ()
                acceptedAcc = lastStageAcc

                permittedAccDrop = 2.0

                # The following if block is just for ease of reproducability of results presented in the paper. 
                # Different values of permittedAccDrop can be tried out for all algorithms, datasets

                if self.algo == config.Algo.rnn:
                    permittedAccDrop = 2.0
                elif self.algo == config.Algo.protonn:
                    if self.dataset in ["curet-multiclass", "letter-multiclass", "ward-binary"]:
                        permittedAccDrop = 2.0
                    else:
                        permittedAccDrop = 1.0
                elif self.algo == config.Algo.bonsai:
                    if self.dataset == "curet-multiclass":
                        permittedAccDrop = 2.0
                    else:
                        permittedAccDrop = 1.0


                for ((demotedVars, _), metrics) in self.varDemoteDetails:
                    acc = metrics[0]
                    if (self.flAccuracy - acc) > permittedAccDrop:
                        break
                    else:
                        okToDemote = demotedVars
                        acceptedAcc = acc
                
                self.demotedVarsList = [i for i in okToDemote] + [i for i in self.demotedVarsList]
                self.demotedVarsOffsets.update(demotedVarsListToOffsets.get(okToDemote, {}))

                if acceptedAcc != lastStageAcc:
                    lastStageAcc = acceptedAcc
                else:
                    print("No difference in iteration %d's stages 1 & 2. Stopping search."%fixedPointCounter)
                    break


            if not config.vbwEnabled or not config.fixedPointVbwIteration:
                break

        return True

    # Reverse sort the accuracies, print the top 5 accuracies and return the
    # best scaling factor
    def getBestScale(self):
        def getMaximisingMetricValue(a):
            if self.maximisingMetric == config.MaximisingMetric.accuracy:
                return (a[1][0], -a[1][1], -a[1][2]) if not config.higherOffsetBias else (a[1][0], -a[0])
            elif self.maximisingMetric == config.MaximisingMetric.disagreements:
                return (-a[1][1], -a[1][2], a[1][0]) if not config.higherOffsetBias else (-max(5, a[1][1]), -a[0])
            elif self.maximisingMetric == config.MaximisingMetric.reducedDisagreements:
                return (-a[1][2], -a[1][1], a[1][0]) if not config.higherOffsetBias else (-max(5, a[1][2]), -a[0])
        x = [(i, self.accuracy[i]) for i in self.accuracy]
        x.sort(key=getMaximisingMetricValue, reverse=True)
        sorted_accuracy = x[:5]
        print(sorted_accuracy)
        return sorted_accuracy[0][0]

    # Find the scaling factor which works best on the training dataset and
    # predict on the testing dataset
    def findBestScalingFactor(self):
        print("-------------------------------------------------")
        print("Performing search to find the best scaling factor")
        print("-------------------------------------------------\n")

        # Generate input files for training dataset
        res = self.convert(config.Version.fixed,
                           config.DatasetType.training, config.Target.x86)
        if res == False:
            return False

        # Search for the best scaling factor
        res = self.performSearch()
        if res == False:
            return False

        print("Best scaling factor = %d" % (self.sf))

        return True

    def runOnTestingDataset(self):
        print("\n-------------------------------")
        print("Prediction on testing dataset")
        print("-------------------------------\n")

        print("Setting max scaling factor to %d\n" % (self.sf))

        if config.vbwEnabled:
            print("Demoted Vars with Offsets: %s\n" % (str(self.demotedVarsOffsets)))

        # Generate files for the testing dataset
        res = self.convert(config.Version.fixed,
                           config.DatasetType.testing, config.Target.x86)
        if res == False:
            return False

        # Compile and run code using the best scaling factor
        if config.vbwEnabled:
            compiled = self.partialCompile(config.Version.fixed, config.Target.x86, self.sf, True, None, 0, dict(self.variableToBitwidthMap), list(self.demotedVarsList), dict(self.demotedVarsOffsets))
        else:
            compiled = self.partialCompile(config.Version.fixed, config.Target.x86, self.sf, True, None, 0)
        if compiled == False:
            return False
            
        res, exit = self.runAll(config.Version.fixed, config.DatasetType.testing, {"default" : self.sf})
        if res == False:
            return False

        return True

    # Generate files for training dataset and perform a profiled execution
    def collectProfileData(self):
        print("-----------------------")
        print("Collecting profile data")
        print("-----------------------")

        res = self.convert(config.Version.floatt,
                           config.DatasetType.training, config.Target.x86)
        if res == False:
            return False

        res = self.compile(config.Version.floatt, config.Target.x86, self.sf)
        if res == False:
            return False

        execMap = self.predict(config.Version.floatt, config.DatasetType.training)
        if execMap == None:
            return False

        self.flAccuracy = execMap["default"][0]
        print("Accuracy is %.3f%%\n" % (execMap["default"][0]))

    # Generate code for Arduino
    def compileFixedForTarget(self):
        print("------------------------------")
        print("Generating code for %s..." % (self.target))
        print("------------------------------\n")

        demotedVarsOffsets = dict(self.demotedVarsOffsets) if hasattr(self, 'demotedVarsOffsets') else {}
        variableToBitwidthMap = dict(self.variableToBitwidthMap) if hasattr(self, 'variableToBitwidthMap') else {}
        res = self.convert(config.Version.fixed,
                           config.DatasetType.testing, self.target, variableToBitwidthMap, demotedVarsOffsets)
        if res == False:
            return False

        # Copy file
        srcFile = os.path.join(config.outdir, "input", "model_fixed.h")
        destFile = os.path.join(config.outdir, str(config.wordLength), self.algo, self.dataset, "model.h")
        os.makedirs(os.path.join(config.outdir, str(config.wordLength), self.algo, self.dataset), exist_ok=True)
        shutil.copyfile(srcFile, destFile)

        # Copy library.h file
        curr_dir = os.path.dirname(os.path.realpath(__file__))

        srcFile = os.path.join(curr_dir, self.target, "library", "library_fixed.h")
        destFile = os.path.join(config.outdir, str(config.wordLength), self.algo, self.dataset, "library.h")
        shutil.copyfile(srcFile, destFile)


        modifiedBitwidths = dict.fromkeys(self.variableToBitwidthMap.keys(), config.wordLength) if hasattr(self, 'variableToBitwidthMap') else {}
        if hasattr(self, 'demotedVarsList'):
            for i in self.demotedVarsList:
                modifiedBitwidths[i] = config.wordLength // 2
        res = self.partialCompile(config.Version.fixed, self.target, self.sf, True, None, 0, dict(modifiedBitwidths), list(self.demotedVarsList) if hasattr(self, 'demotedVarsList') else [], dict(demotedVarsOffsets))
        if res == False:
            return False

        return True

    def runForFixed(self):
        # Collect runtime profile
        res = self.collectProfileData()
        if res == False:
            return False

        # Obtain best scaling factor
        if self.sf == None:
            res = self.findBestScalingFactor()
            if res == False:
                return False

        res = self.runOnTestingDataset()
        if res == False:
            return False
        else:
            self.testingAccuracy = self.accuracy[self.sf]

        # Generate code for target
        if self.target == config.Target.arduino:
            self.compileFixedForTarget()

            print("\nArduino sketch dumped in the folder %s\n" % (config.outdir))

        return True

    def compileFloatForTarget(self):
        print("------------------------------")
        print("Generating code for %s..." % (self.target))
        print("------------------------------\n")

        res = self.convert(config.Version.floatt,
                           config.DatasetType.testing, self.target)
        if res == False:
            return False

        res = self.compile(config.Version.floatt, self.target, self.sf)
        if res == False:
            return False

        # Copy model.h
        srcFile = os.path.join(config.outdir, "input", "model_float.h")	
        destFile = os.path.join(config.outdir, "float", self.algo, self.dataset, "model.h")	
        os.makedirs(os.path.join(config.outdir, "float", self.algo, self.dataset), exist_ok=True)
        shutil.copyfile(srcFile, destFile)

        # Copy library.h file
        curr_dir = os.path.dirname(os.path.realpath(__file__))	
        srcFile = os.path.join(curr_dir, self.target, "library", "library_float.h")	
        destFile = os.path.join(config.outdir, "float", self.algo, self.dataset, "library.h")
        shutil.copyfile(srcFile, destFile)

        return True

    def runForFloat(self):
        print("---------------------------")
        print("Executing for X86 target...")
        print("---------------------------\n")

        res = self.convert(config.Version.floatt,
                           config.DatasetType.testing, config.Target.x86)
        if res == False:
            return False

        res = self.compile(config.Version.floatt, config.Target.x86, self.sf)
        if res == False:
            return False

        execMap = self.predict(config.Version.floatt, config.DatasetType.testing)
        if execMap == None:
            return False
        else:
            self.testingAccuracy = execMap["default"][0]

        print("Accuracy is %.3f%%\n" % (self.testingAccuracy))

        if self.target == config.Target.arduino:
            self.compileFloatForTarget()
            print("\nArduino sketch dumped in the folder %s\n" % (config.outdir))

        return True

    def run(self):

        sys.setrecursionlimit(10000)

        self.setup()

        if self.version == config.Version.fixed:
            return self.runForFixed()
        else:
            return self.runForFloat()
