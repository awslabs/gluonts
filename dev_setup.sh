#!/usr/bin/env bash

ACT_GIT_VERSION=$(git --version | cut -d' ' -f3)
REQ_GIT_VERSION="2.9"

ACT_PYTHON_VERSION=$(python --version | cut -d' ' -f2)
REQ_PYTHON_VERSION="3.6"

#!/bin/bash
vercomp () {
    if [[ $1 == $2 ]]
    then
        return 1
    fi
    local IFS=.
    local i ver1=($1) ver2=($2)
    # fill empty fields in ver1 with zeros
    for ((i=${#ver1[@]}; i<${#ver2[@]}; i++))
    do
        ver1[i]=0
    done
    for ((i=0; i<${#ver1[@]}; i++))
    do
        if [[ -z ${ver2[i]} ]]
        then
            # fill empty fields in ver2 with zeros
            ver2[i]=0
        fi
        if ((10#${ver1[i]} > 10#${ver2[i]}))
        then
            return 0
        fi
        if ((10#${ver1[i]} < 10#${ver2[i]}))
        then
            return 2
        fi
    done
    return 1
}

vercomp ${ACT_GIT_VERSION} ${REQ_GIT_VERSION}
if [[ $? -gt 1 ]]; then
    echo "Git version ${ACT_GIT_VERSION} detected," \
         "but >= ${REQ_GIT_VERSION} needed in order to set the 'core.hooksPath' configuration variable."
    exit 1
fi

vercomp ${ACT_PYTHON_VERSION} ${REQ_PYTHON_VERSION}
if [[ $? -gt 1 ]]; then
    echo "Python version ${ACT_PYTHON_VERSION} detected," \
         "but >= ${REQ_PYTHON_VERSION} needed in order to run GluonTS."
    exit 1
fi

# update location of Git hooks from default (.git/hooks) to the versioned folder .devtools/githooks
git config core.hooksPath ".devtools/githooks"

# install project requirements
python -m pip install -r requirements/requirements-setup.txt
python -m pip install -r requirements/requirements-test.txt
python -m pip install -r requirements/requirements.txt
