pipeline {
    agent {label 'builder-1'}

    options {
        timeout(time: 15, unit: 'MINUTES')
        timestamps()
    }

    environment {
        REPO_NAME = 'agent-service'
        NEXUS_URL = 'nexus.boosted.ai:8081'
        NEXUS_DEPLOYMENT_CREDS = credentials('03e90d96-4118-4a28-8b7d-7c0f09c85ae4')
        DEPLOYMENTS_REPO_VERSION_NAME = "AgentService"
        DEPLOYMENTS_REPO_BRANCH = "master"
        MASTER_BRANCH = "main"
    }

    stages {
        stage('Checkout') {
            steps {
                script {
                    if (env.BRANCH_NAME == env.MASTER_BRANCH)
                    {
                        withCredentials([gitUsernamePassword(credentialsId: 'GitHub', gitToolName: 'Default')]) {
                                                sh "whoami"
                                                checkout([$class: 'GitSCM', branches: [[name: "*/${env.BRANCH_NAME}"]], extensions: [], userRemoteConfigs: [[credentialsId: 'GitHub', url: "https://github.com/GBI-Core/${REPO_NAME}"]]])
                                                VERSION = sh(script: "./bump_version.sh", returnStdout: true).trim()
                                                sh "echo ${VERSION}"
                                                sh "./build_and_push_image.sh ${VERSION}"
                                                checkout([$class: 'GitSCM', branches: [[name: "*/${DEPLOYMENTS_REPO_BRANCH}"]], extensions: [], userRemoteConfigs: [[credentialsId: 'GitHub', url: "https://github.com/GBI-Core/deployments"]]])
                                                sh "git config user.name ${GIT_USERNAME}"
                                                sh "git config user.email jenkins@boosted.ai"
                                                sh "git status"
                                                sh "./version_bump.sh ${DEPLOYMENTS_REPO_VERSION_NAME} ${VERSION}"
                                                sh "git diff"
                                                sh "git commit -m \"Autobump ${DEPLOYMENTS_REPO_VERSION_NAME} to ${VERSION}\""
                                                sh "git show"
                                                sh "git push origin HEAD:${DEPLOYMENTS_REPO_BRANCH}"
                                                }
                    }
                    // If the commit message contains the string '[build-me]' then build a custom version
                    result = sh(script: "git log -1 --oneline | grep '.*\\[build-me\\].*'", returnStatus: true)
                    if (result == 0) {
                      sh "echo [build-me] detected in commit message. Building image for commit..."
                      sh "./build_and_push_image.sh"
                    }

                    if (env.BRANCH_NAME != env.MASTER_BRANCH){
                        sh "docker build -f test.dockerfile -t \$(git rev-parse HEAD) ."
                        sh "docker run --network host \$(git rev-parse HEAD)"
                    }
                }
            }
        }

    }
}
