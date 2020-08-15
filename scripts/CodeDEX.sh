#!/bin/bash
export FORTIFY_HOME=/usr1/CloudDragon/tools/CodeDEX/plugins/CodeDEX/tool/fortify
export PATH=$FORTIFY_HOME/bin:$PATH
#Fortify build_id necessary
export FORITYF_BUILD_ID=test

#Python Project Root
export ProjectRoot=$WORKSPACE
#Specifies the path for additional import directories
export PythonPath=$WORKSPACE

sourceanalyzer -b $FORITYF_BUILD_ID -clean

#From Fortify16.10, if a python project uses Django Framework, there are some additional options:
#1. In the fortify-sca.properties configuration file:
#   com.fortify.sca.limiters.MaxPassThroughChainDepth=8
#   com.fortify.sca.limiters.MaxChainDepth=8
#2. In the translate command(the following command), add -django-template-dirs <path/to/template/dirs>
sourceanalyzer -b $FORITYF_BUILD_ID $ProjectRoot **/*.py -python-path $PythonPath -encoding UTF-8 # -django-template-dirs <path/to/template/dirs>