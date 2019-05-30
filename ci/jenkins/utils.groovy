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

// initialize source codes
def init_git() {
  deleteDir()
  retry(5) {
    try {
      // Make sure wait long enough for api.github.com request quota. Important: Don't increase the amount of
      // retries as this will increase the amount of requests and worsen the throttling
      timeout(time: 15, unit: 'MINUTES') {
        checkout scm
        sh 'git clean -xdff'
        sh 'git reset --hard'
        sh 'git submodule update --init --recursive'
        sh 'git submodule foreach --recursive git clean -ffxd'
        sh 'git submodule foreach --recursive git reset --hard'
      }
    } catch (exc) {
      deleteDir()
      error "Failed to fetch source codes with ${exc}"
      sleep 2
    }
  }
}


def get_git_commit_hash() {
  lastCommitMessage = sh (script: "git log -1 --pretty=%B", returnStdout: true)
  lastCommitMessage = lastCommitMessage.trim()
  if (lastCommitMessage.startsWith("Merge commit '") && lastCommitMessage.endsWith("' into HEAD")) {
      // Merge commit applied by Jenkins, skip that commit
      git_commit_hash = sh (script: "git rev-parse @~", returnStdout: true)
  } else {
      git_commit_hash = sh (script: "git rev-parse @", returnStdout: true)
  }
  return git_commit_hash.trim()
}

def publish_test_coverage(codecov_credential) {
    // CodeCovs auto detection has trouble with our CIs PR validation due the merging strategy
    git_commit_hash = get_git_commit_hash()

    if (env.CHANGE_ID) {
      // PR execution
      codecovArgs = "-B ${env.CHANGE_TARGET} -C ${git_commit_hash} -P ${env.CHANGE_ID}"
    } else {
      // Branch execution
      codecovArgs = "-B ${env.BRANCH_NAME} -C ${git_commit_hash}"
    }

    // To make sure we never fail because test coverage reporting is not available
    // Fall back to our own copy of the bash helper if it failed to download the public version
    withCredentials([string(credentialsId: codecov_credential, variable: 'CODECOV_TOKEN')]) {
      sh "(curl --retry 10 -s https://codecov.io/bash | bash -s - ${codecovArgs}) || (curl --retry 10 -s https://s3-us-west-2.amazonaws.com/mxnet-ci-prod-slave-data/codecov-bash.txt | bash -s - ${codecovArgs}) || true"
    }
}

// Allow publishing to GitHub with a custom context (the status shown under a PR)
// Credit to https://plugins.jenkins.io/github
def get_repo_url() {
  checkout scm
  return sh(returnStdout: true, script: "git config --get remote.origin.url").trim()
}

def update_github_commit_status(state, message) {
  node {
    // NOTE: https://issues.jenkins-ci.org/browse/JENKINS-39482
    //The GitHubCommitStatusSetter requires that the Git Server is defined under
    //*Manage Jenkins > Configure System > GitHub > GitHub Servers*.
    //Otherwise the GitHubCommitStatusSetter is not able to resolve the repository name
    //properly and you would see an empty list of repos:
    //[Set GitHub commit status (universal)] PENDING on repos [] (sha:xxxxxxx) with context:test/mycontext
    //See https://cwiki.apache.org/confluence/display/MXNET/Troubleshooting#Troubleshooting-GitHubcommit/PRstatusdoesnotgetpublished

    echo "Publishing commit status..."

    repoUrl = get_repo_url()
    echo "repoUrl=${repoUrl}"

    commitSha = get_git_commit_hash()
    echo "commitSha=${commitSha}"

    context = get_github_context()
    echo "context=${context}"

    // a few attempts need to be made: https://github.com/apache/incubator-mxnet/issues/11654
    for (int attempt = 1; attempt <= 3; attempt++) {
      echo "Sending GitHub status attempt ${attempt}..."

      step([
        $class: 'GitHubCommitStatusSetter',
        reposSource: [$class: "ManuallyEnteredRepositorySource", url: repoUrl],
        contextSource: [$class: "ManuallyEnteredCommitContextSource", context: context],
        commitShaSource: [$class: "ManuallyEnteredShaSource", sha: commitSha],
        statusBackrefSource: [$class: "ManuallyEnteredBackrefSource", backref: "${env.RUN_DISPLAY_URL}"],
        errorHandlers: [[$class: 'ShallowAnyErrorHandler']],
        statusResultSource: [
          $class: 'ConditionalStatusResultSource',
          results: [[$class: "AnyBuildResult", message: message, state: state]]
        ]
      ])

      if (attempt <= 2) {
        sleep 1
      }
    }

    echo "Publishing commit status done."

  }
}

def get_github_context() {
  // Since we use multi-branch pipelines, Jenkins appends the branch name to the job name
  if (env.BRANCH_NAME) {
    short_job_name = JOB_NAME.substring(0, JOB_NAME.lastIndexOf('/'))
  } else {
    short_job_name = JOB_NAME
  }

  return "ci/jenkins/${short_job_name}"
}

def parallel_stage(stage_name, steps) {
    // Allow to pass an array of steps that will be executed in parallel in a stage
    new_map = [:]

    for (def step in steps) {
        new_map = new_map << step
    }

    stage(stage_name) {
      parallel new_map
    }
}

def assign_node_labels(args) {
  // This function allows to assign instance labels to the generalized placeholders.
  // This serves two purposes:
  // 1. Allow generalized placeholders (e.g. NODE_WINDOWS_CPU) in the job definition
  //    in order to abstract away the underlying node label. This allows to schedule a job
  //    onto a different node for testing or security reasons. This could be, for example,
  //    when you want to test a new set of slaves on separate labels or when a job should
  //    only be run on restricted slaves
  // 2. Restrict the allowed job types within a Jenkinsfile. For example, a UNIX-CPU-only
  //    Jenkinsfile should not allowed access to Windows or GPU instances. This prevents
  //    users from just copy&pasting something into an existing Jenkinsfile without
  //    knowing about the limitations.
  NODE_LINUX_GPU = args.linux_gpu
  NODE_LINUX_CPU = args.linux_cpu
}

def main_wrapper(args) {
  // Main Jenkinsfile pipeline wrapper handler that allows to wrap core logic into a format
  // that supports proper failure handling
  // args:
  // - core_logic: Jenkins pipeline containing core execution logic
  // - failure_handler: Failure handler

  // assign any caught errors here
  err = null
  try {
    update_github_commit_status('PENDING', 'Job has been enqueued')
    args['core_logic']()

    // set build status to success at the end
    currentBuild.result = "SUCCESS"
    update_github_commit_status('SUCCESS', 'Job succeeded')
  } catch (caughtError) {
    node {
      sh "echo caught ${caughtError}"
      err = caughtError
      currentBuild.result = "FAILURE"
      update_github_commit_status('FAILURE', 'Job failed')
    }
  } finally {
    node {
      // Call failure handler
      args['failure_handler']()

      // Clean workspace to reduce space requirements
      cleanWs()

      // Remember to rethrow so the build is marked as failing
      if (err) {
        throw err
      }
    }
  }
}

return this
