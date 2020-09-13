# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import numpy as np
import os

from seedot.compiler.codegen.codegenBase import CodegenBase

import seedot.compiler.ir.ir as IR
import seedot.compiler.ir.irUtil as IRUtil

import seedot.compiler.type as Type
from seedot.util import *
from seedot.writer import Writer

import time

class X86(CodegenBase):

    def __init__(self, outputDir, generateAllFiles, printSwitch, idStr, decls, localDecls, scales, intvs, cnsts, expTables, globalVars, internalVars, floatConstants, substitutions, demotedVarsOffsets, varsForBitwidth, varLiveIntervals, notScratch):
        self.outputDir = outputDir
        cppFile = os.path.join(
            self.outputDir, "seedot_" + getVersion() + ".cpp")
        if generateAllFiles:
            self.out = Writer(cppFile)
        else:
            if debugCompiler():
                print("Opening file to output cpp code: ID" + idStr)
            for i in range(3):
                if debugCompiler():
                    print("Try %d" % (i+1))
                try:
                    self.out = Writer(cppFile, "a")
                except:
                    if debugCompiler():
                        print("OS prevented file from opening. Sleeping for %d seconds" % (i+1))
                    time.sleep(i+1)
                else:
                    if debugCompiler():
                        print("Opened")
                    break

        self.decls = decls
        self.localDecls = localDecls
        self.scales = scales
        self.intvs = intvs
        self.cnsts = cnsts
        self.expTables = expTables
        self.globalVars = globalVars
        self.internalVars = internalVars
        self.floatConstants = floatConstants

        self.generateAllFiles = generateAllFiles
        self.idStr = idStr
        self.printSwitch = printSwitch

        self.demotedVarsOffsets = demotedVarsOffsets
        self.varsForBitwidth = varsForBitwidth

        self.varLiveIntervals = varLiveIntervals
        self.notScratch = notScratch

    def printPrefix(self):

        if self.generateAllFiles:
            self.printCincludes()

            self.printExpTables()

            self.printVarDecls()

        self.printCHeader()

        self.printModelParamsWithBitwidth()

        self.printVarDecls(globalVarDecl=False)

        self.printConstDecls()

        self.out.printf('\n')

    def printCincludes(self):
        self.out.printf('#include <iostream>\n', indent=True)
        self.out.printf('#include <cstring>\n', indent=True)
        self.out.printf('#include <cmath>\n\n', indent=True)
        self.out.printf('#include "datatypes.h"\n', indent=True)
        self.out.printf('#include "predictors.h"\n', indent=True)
        self.out.printf('#include "profile.h"\n', indent=True)
        self.out.printf('#include "library_%s.h"\n' %
                        (getVersion()), indent=True)
        self.out.printf('#include "model_%s.h"\n' %
                        (getVersion()), indent=True)
        self.out.printf('#include "vars_%s.h"\n\n' %
                        (getVersion()), indent=True)
        self.out.printf('using namespace std;\n', indent=True)
        self.out.printf('using namespace seedot_%s;\n' %
                        (getVersion()), indent=True)
        # self.out.printf('using namespace vars_%s;\n\n' %
        #                 (getVersion()), indent=True)

    def printExpTables(self):
        for exp, [table, [tableVarA, tableVarB]] in self.expTables.items():
            self.printExpTable(table[0], tableVarA)
            self.printExpTable(table[1], tableVarB)
            self.out.printf('\n')

    def printExpTable(self, table_row, var):
        self.out.printf('const MYINT %s[%d] = {\n' % (
            var.idf, len(table_row)), indent=True)
        self.out.increaseIndent()
        self.out.printf('', indent=True)
        for i in range(len(table_row)):
            self.out.printf('%d, ' % table_row[i])
        self.out.decreaseIndent()
        self.out.printf('\n};\n')

    def printCHeader(self):
        if forFloat():
            func = "Float"
            type = "float"
        else:
            func = "Fixed"
            type = "MYINT"
        if forFloat():
            self.out.printf('int seedot%s(%s **X) {\n' % (func, type), indent=True)
        else: 
            self.out.printf('int seedot%s%s(%s **X%s) {\n' % (func, self.idStr if not self.generateAllFiles else "", type, "_temp" if config.vbwEnabled else ""), indent=True)
        self.out.increaseIndent()

    def printModelParamsWithBitwidth(self):
        if config.vbwEnabled and forFixed():
            for var in self.globalVars:
                if var + "idx" in self.globalVars and var + "val" in self.globalVars:
                    continue
                bw = self.varsForBitwidth[var]
                typ_str = "int%d_t" % bw
                size = self.decls[var].shape
                sizestr = ''.join(["[%d]" % (i) for i in size])

                Xindexstr = ''
                Xintstar = ''.join(["*" for i in size])

                if var != 'X':
                    self.out.printf(typ_str + " " + var + sizestr + ";\n", indent = True)
                else:
                    self.out.printf(typ_str + Xintstar + " " + var + ";\n", indent = True)

                for i in range(len(size)):
                    Xindexstr += ("[i" + str(i-1) + "]" if i > 0 else "")
                    if var == 'X':
                        Xintstar = Xintstar[:-1]
                        self.out.printf("X%s = new %s%s[%d];\n" % (Xindexstr, typ_str, Xintstar, size[i]), indent=True)
                    self.out.printf("for (int i%d = 0; i%d < %d; i%d ++) {\n" % (i,i,size[i], i), indent = True)
                    self.out.increaseIndent()

                indexstr = ''.join("[i" + str(i) + "]" for i in range(len(size)))
                divide = int(round(np.ldexp(1, config.wordLength - self.varsForBitwidth[var] + (self.demotedVarsOffsets.get(var, 0) if self.varsForBitwidth[var] != config.wordLength else 0) ))) if var[-3:] != "idx" and var != "X" else 1
                self.out.printf(var + indexstr + " = " + var + "_temp" + indexstr + "/" + str(divide) + ";\n", indent = True)

                for i in range(len(size)):
                    self.out.decreaseIndent()
                    self.out.printf("}\n", indent = True)

    def printVarDecls(self, globalVarDecl=True):
        if self.generateAllFiles:
            varsFilePath = os.path.join(
                self.outputDir, "vars_" + getVersion() + ".h")
            varsFile = Writer(varsFilePath)

            varsFile.printf("#pragma once\n\n")
            varsFile.printf("#include \"datatypes.h\"\n\n")
            varsFile.printf("namespace vars_%s {\n" % (getVersion()))
            varsFile.increaseIndent()

        for decl in self.decls:
            if decl in self.globalVars:
                continue

            if forFloat() and decl not in self.internalVars:
                typ_str = IR.DataType.getFloatStr()
            elif forFixed() and decl not in self.internalVars:
                if config.vbwEnabled and decl not in self.internalVars:
                    bw = self.varsForBitwidth.get(decl, config.wordLength)
                    typ_str = "int%d_t" % bw
                else:
                    typ_str = IR.DataType.getIntStr()
            else:
                typ_str = IR.DataType.getIntStr()

            idf_str = decl
            type = self.decls[decl]
            if Type.isInt(type):
                shape_str = ''
            elif Type.isTensor(type):
                shape_str = ''.join(['[' + str(n) + ']' for n in type.shape])

            if not config.vbwEnabled:
                self.out.printf('%s vars_%s::%s%s;\n', typ_str,
                                getVersion(), idf_str, shape_str, indent=True)
                if self.generateAllFiles:
                    varsFile.printf('extern %s %s%s;\n', typ_str,
                                idf_str, shape_str, indent=True)
            else:
                if forFixed() and idf_str in self.varsForBitwidth and idf_str[:3] == "tmp":
                    if globalVarDecl:
                        for bw in config.availableBitwidths:
                            self.out.printf("int%d_t vars_%s::%s_%d%s;\n", bw, getVersion(), idf_str, bw, shape_str, indent=True)
                    else:
                        self.out.printf("int%d_t %s_%d%s;\n", self.varsForBitwidth[idf_str], idf_str, bw, shape_str, indent=True)
                else:
                    if globalVarDecl:
                        self.out.printf("%s vars_%s::%s%s;\n", typ_str, getVersion(), idf_str, shape_str, indent=True)
                    else:
                        self.out.printf("%s %s%s;\n", typ_str, idf_str, shape_str, indent=True)

                if self.generateAllFiles:
                    if forFixed() and idf_str in self.varsForBitwidth and idf_str[:3] == "tmp":
                        for bw in config.availableBitwidths:
                            varsFile.printf("extern int%d_t %s_%d%s;\n", bw, idf_str, bw, shape_str, indent=True)
                    else:
                        varsFile.printf("extern %s %s%s;\n", typ_str, idf_str, shape_str, indent=True)

        self.out.printf('\n')
        if self.generateAllFiles:
            varsFile.decreaseIndent()
            varsFile.printf("}\n")
            varsFile.close()

        self.generateDebugProgram()

    def generateDebugProgram(self):
        if not self.generateAllFiles:
            return
        debugFilePath = os.path.join(self.outputDir, "debug.cpp")
        debugFile = Writer(debugFilePath)

        debugFile.printf("#include <iostream>\n\n")
        debugFile.printf("#include \"datatypes.h\"\n")
        debugFile.printf("#include \"profile.h\"\n")
        debugFile.printf("#include \"vars_fixed.h\"\n")
        debugFile.printf("#include \"vars_float.h\"\n\n")
        debugFile.printf("using namespace std;\n\n")
        debugFile.printf("void debug() {\n\n")

        if debugMode() and forFixed():
            debugFile.increaseIndent()

            for decl in self.decls:
                if decl in self.globalVars:
                    continue

                type = self.decls[decl]
                if decl not in self.scales or not isinstance(type, Type.Tensor) or type.isShapeOne():
                    continue

                scale = self.scales[decl]

                s = decl + "[0]" * type.dim
                shape_str = ''.join([str(n) + ', ' for n in type.shape])
                shape_str = shape_str.rstrip(', ')

                debugFile.printf("diff(&vars_float::%s, &vars_fixed::%s, %d, %s);\n\n" % (
                    s, s, scale, shape_str), indent=True)

            debugFile.decreaseIndent()

        debugFile.printf("}\n")

        debugFile.close()

    def printSuffix(self, expr: IR.Expr):
        self.out.printf('\n')

        if config.vbwEnabled and forFixed():
            bw = self.varsForBitwidth['X']
            typ_str = "int%d_t" % bw
            size = self.decls['X'].shape
            sizestr = ''.join([("[%d]" % i) for i in size])

            Xindexstr = ''
            Xintstar = ''.join(["*" for i in size])

            for i in range(len(size)):
                Xindexstr += (("[i%d]" % (i-1)) if i > 0 else "")
                self.out.printf("for (int i%d = 0; i%d < %d; i%d ++ ){\n" % (i,i,size[i],i), indent=True)
                self.out.increaseIndent()

            for i in range(len(size)-1, -1, -1):
                self.out.decreaseIndent()
                self.out.printf("}\n", indent=True)
                self.out.printf("delete[] X%s;\n" % (Xindexstr), indent=True)
                Xindexstr = Xindexstr[:-4] if len(Xindexstr) > 0 else Xindexstr
                assert len(size) < 10, "Too simple logic for printing indices used, cannot handle 10+ Dim Tensors"


        type = self.decls[expr.idf]

        if Type.isInt(type):
            self.out.printf('return ', indent=True)
            self.print(expr)
            self.out.printf(';\n')
        elif Type.isTensor(type):
            idfr = expr.idf
            exponent = self.scales[expr.idf]
            num = 2 ** exponent

            if type.dim == 0:
                self.out.printf('cout << ', indent=True)
                self.out.printf('float(' + idfr + ')*' + str(num))
                self.out.printf(' << endl;\n')
            else:
                iters = []
                for i in range(type.dim):
                    s = chr(ord('i') + i)
                    tempVar = IR.Var(s)
                    iters.append(tempVar)
                expr_1 = IRUtil.addIndex(expr, iters)
                cmds = IRUtil.loop(type.shape, iters, [
                                   IR.PrintAsFloat(expr_1, exponent)])
                self.print(IR.Prog(cmds))
        else:
            assert False

        self.out.decreaseIndent()
        self.out.printf('}\n', indent=True)

        def isInt(a):
            try:
                int(a)
                return True
            except:
                return False

        if forFixed():
            if (int(self.printSwitch) if isInt(self.printSwitch) else -2) > -1:
                self.out.printf("const int switches = %d;\n" % (int(self.printSwitch)), indent = True)
                self.out.printf('void seedotFixedSwitch(int i, MYINT **X_temp, int& res) {\n', indent=True)
                self.out.increaseIndent()
                self.out.printf('switch(i) {\n', indent = True)
                self.out.increaseIndent()
                for i in range(int(self.printSwitch)):
                    self.out.printf('case %d: res = seedotFixed%d(X_temp); return;\n' % (i,i+1), indent = True)
                self.out.printf('default: res = -1; return;\n', indent = True)
                self.out.decreaseIndent()
                self.out.printf('}\n', indent=True)
                self.out.decreaseIndent()
                self.out.printf('}\n', indent=True)
        if debugCompiler():
            print("Closing File after outputting cpp code: ID " + self.idStr)
        self.out.close()
