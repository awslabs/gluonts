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
    uv run python -m ipykernel install --user --name docsbuild
    uv run python {{MD2IPYNB}} --kernel docsbuild docs/tutorials/**/*.md.template --mode {{mode}}

release:
  uv build

black:
  uv run black --check --diff --color src test examples

license:
  uv run python .devtools/license check src test

# Install the package with all development dependencies
install-dev:
  uv sync --all-extras

# Run tests
test *args:
  uv run pytest -n2 --doctest-modules --ignore test/nursery test {{args}}

# Run linting checks
lint-check:
  uv run ruff check src test

# Run type checks
type-check:
  uv run mypy src

# Run format checks
format-check:
  uv run ruff format --check src test

# Format code
format:
  uv run ruff format src test
