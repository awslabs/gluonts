# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

ROOTDIR := justfile_directory()
MD2IPYNB := ROOTDIR + "/docs/md2ipynb.py"

mode := "release"

docs: compile_notebooks
  make -C docs html # SPHINXOPTS=-W

clean:
  git clean -ff -d -x --exclude="{{ROOTDIR}}/tests/externaldata/*" --exclude="{{ROOTDIR}}/tests/data/*" --exclude="{{ROOTDIR}}/conda/"

compile_notebooks:
    python -m ipykernel install --user --name docsbuild
    python {{MD2IPYNB}} --kernel docsbuild docs/tutorials/**/*.md.template --mode {{mode}}

release:
  python setup.py sdist

black:
  black --check --diff --color src test examples

mypy:
  python setup.py type_check

license:
  python .devtools/license check src test
