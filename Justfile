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

ROOTDIR := invocation_directory()
MD2IPYNB := ROOTDIR + "/docs/md2ipynb.py"

skip_build_notebook := env_var_or_default("SKIP_BUILD_NOTEBOOK", "false")


docs: release
  make -C docs html # SPHINXOPTS=-W

clean:
  git clean -ff -d -x --exclude="{{ROOTDIR}}/tests/externaldata/*" --exclude="{{ROOTDIR}}/tests/data/*" --exclude="{{ROOTDIR}}/conda/"

compile_notebooks:
  if [ {{skip_build_notebook}} = "true" ] ; \
  then \
    find docs/tutorials/**/*.md.input | sed 'p;s/\.input//' | xargs -n2 cp; \
  else \
    python -m ipykernel install --user --name docsbuild; \
    python {{MD2IPYNB}} --kernel docsbuild "docs/tutorials/**/*.md.input"; \
  fi;

dist_notebooks: compile_notebooks
  cd docs/tutorials && \
  find * -type d -prune | grep -v 'tests\|__pycache__' | xargs -t -n 1 -I{} zip --no-dir-entries -r {}.zip {} -x "*.md" -x "__pycache__" -x "*.pyc" -x "*.txt" -x "*.log" -x "*.params" -x "*.npz" -x "*.json"

release: dist_notebooks
  python setup.py sdist
