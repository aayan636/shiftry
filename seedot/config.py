# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

# Target word length. Currently set to match the word length of Arduino (2 bytes)
wordLength = 16
availableBitwidths = [8, 16, 32]

# Range of max scale factor used for exploration
maxScaleRange = 0, -wordLength

# tanh approximation limit
tanhLimit = 1.0

# MSBuild location
# Edit the path if not present at the following location
msbuildPathOptions = [r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\MSBuild\Current\Bin\MSBuild.exe",
                      r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise\MSBuild\Current\Bin\MSBuild.exe",
                      r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\MSBuild\Current\Bin\MSBuild.exe"
                      ]

ddsEnabled = True
vbwEnabled = True
functionReducedProfiling = True

trimHighestDecile = False

higherOffsetBias = True

fixedPointVbwIteration = False

class MaximisingMetric:
    accuracy = "acc"
    disagreements = "disagree"
    reducedDisagreements = "red_disagree"
    default = [accuracy]
    all = [accuracy, disagreements, reducedDisagreements]

class Algo:
    bonsai = "bonsai"
    lenet = "lenet"
    protonn = "protonn"
    rnn = "rnn"
    default = [bonsai, protonn]
    all = [bonsai, lenet, protonn, rnn]


class Version:
    fixed = "fixed"
    floatt = "float"
    default = [fixed, floatt]
    all = default


class DatasetType:
    training = "training"
    testing = "testing"
    default = testing
    all = [training, testing]


class Target:
    arduino = "arduino"
    x86 = "x86"
    default = x86
    all = [arduino, x86]
