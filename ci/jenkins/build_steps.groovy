// -*- mode: groovy -*-

// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.
//
// This file contains the steps that will be used in the
// Jenkins pipelines

utils = load('ci/jenkins/utils.groovy')

def sanity_checks(workspace_name, conda_env_name, path) {
  return ['Lint': {
    node {
      ws("workspace/${workspace_name}") {
        timeout(time: max_time, unit: 'MINUTES') {
          utils.init_git()
          sh """
          set -ex
          source ci/prepare_clean_env.sh ${conda_env_name}
          python setup.py style_check
          python setup.py type_check
          python .devtools/license check src test
          set +ex
          """
        }
      }
    }
  }]
}

def test_unittest(workspace_name, conda_env_name,
                  test_path, cov_path,
                  mark,
                  threads, gpu, skip_report) {
  capture_flag = env.BRANCH_NAME.startsWith('PR-')?'':'--capture=no'
  node_type = gpu?NODE_LINUX_GPU:NODE_LINUX_CPU
  return ["${conda_env_name}: ${test_path} -m '${mark}'": {
    node(node_type) {
      ws("workspace/${workspace_name}") {
        timeout(time: max_time, unit: 'MINUTES') {
          utils.init_git()
          sh """
          set -ex
          source ci/prepare_clean_env.sh ${conda_env_name}
          pytest -v ${capture_flag} -n ${threads} -m '${mark}' --durations=30 --cov ${cov_path} --cov-report=term --cov-report xml ${test_path}
          set +ex
          """
          if (!skip_report) utils.publish_test_coverage('GluonTSCodeCov')
        }
      }
    }
  }]
}

def test_doctest(workspace_name, conda_env_name,
                 test_path, cov_path, threads) {
  capture_flag = env.BRANCH_NAME.startsWith('PR-')?'':'--capture=no'
  return ["${conda_env_name}: doctest ${test_path}'": {
    node(NODE_LINUX_CPU) {
      ws("workspace/${workspace_name}") {
        timeout(time: max_time, unit: 'MINUTES') {
          utils.init_git()
          sh """
          set -ex
          source ci/prepare_clean_env.sh ${conda_env_name}
          pytest -v ${capture_flag} -n ${threads} --durations=30 --cov ${cov_path} --cov-report=term --cov-report xml --doctest-modules ${test_path}
          set +ex
          """
          utils.publish_test_coverage('GluonTSCodeCov')
        }
      }
    }
  }]
}

def create_website(workspace_name, conda_env_name) {
  if (env.BRANCH_NAME.startsWith('PR-')){
    bucket = 'gluon-ts-staging'
    path = env.BRANCH_NAME+'/'+env.BUILD_NUMBER
  } else {
    bucket = 'gluon-ts'
    path = env.BRANCH_NAME
  }
  return ["${conda_env_name}: website'": {
    node(NODE_LINUX_GPU) {
      ws("workspace/${workspace_name}") {
        timeout(time: max_time, unit: 'MINUTES') {
          utils.init_git()
          sh """
          set -ex
          source ci/prepare_clean_env.sh ${conda_env_name}
          make docs

          ci/upload_doc.sh ${bucket} ${path}
          set +ex
          """
        }
      }
    }
  }]
}

def website_linkcheck(workspace_name, conda_env_name) {
  // TODO: re-enable to enforce no-WARNING doc builds on master
  // enforce_linkcheck = env.BRANCH_NAME.startsWith('PR-')?'false':'true'
  enforce_linkcheck = 'false'

  return ["${conda_env_name}: website link check'": {
    node(NODE_LINUX_CPU) {
      ws("workspace/${workspace_name}") {
        timeout(time: max_time, unit: 'MINUTES') {
          utils.init_git()
          sh """
          set -ex
          source ci/prepare_clean_env.sh ${conda_env_name}
          if [[ ${enforce_linkcheck} == true ]]; then
              make -C docs linkcheck SPHINXOPTS=-W
          else
              set +e
              make -C docs linkcheck
          fi;
          set +ex
          """
        }
      }
    }
  }]
}

def post_website_link() {
  return ["Deploy: ": {
    node {
      timeout(time: max_time, unit: 'MINUTES') {
        if (env.BRANCH_NAME.startsWith("PR-")) {
            pullRequest.comment("Job ${env.BRANCH_NAME}/${env.BUILD_NUMBER} is complete. \nDocs are uploaded to http://gluon-ts-staging.s3-accelerate.dualstack.amazonaws.com/${env.BRANCH_NAME}/${env.BUILD_NUMBER}/index.html")
        }
      }
    }
  }]
}

return this
