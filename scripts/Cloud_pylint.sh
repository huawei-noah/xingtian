#!/bin/bash
pylint --generate-rcfile > pylint.conf
pylint --rcfile=pylint.conf  --output-format=parseable $1 > $2
exit 0
