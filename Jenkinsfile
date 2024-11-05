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
                                                sh "./build_and_push_test_image.sh ${VERSION}"
                                                sh "./run_regression.sh ${VERSION}"
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
                      sh "./build_and_push_test_image.sh"
                    }

                    if (env.BRANCH_NAME != env.MASTER_BRANCH){
                        withCredentials([gitUsernamePassword(credentialsId: 'GitHub', gitToolName: 'Default')]) {
                        checkout([$class: 'GitSCM', branches: [[name: "*/${env.BRANCH_NAME}"]], extensions: [], userRemoteConfigs: [[credentialsId: 'GitHub', url: "https://github.com/GBI-Core/${REPO_NAME}"]]])
                        sh "echo =============================================="
                        sh "echo begin 'porcelain' prompt diff"
                        sh "echo + = added, - = removed ~ = newline"
                        sh "echo =============================================="
                        sh "git diff --word-diff=porcelain origin/main"
                        sh "echo =============================================="
                        sh "echo end 'porcelain' prompt diff"
                        sh "echo =============================================="

                        sh "echo =============================================="
                        sh "echo begin 'plain' prompt diff"
                        sh "echo look for [+added+][-removed-]"
                        sh "echo =============================================="
                        sh "git diff --word-diff=plain origin/main"
                        sh "echo =============================================="
                        sh "echo end 'plain' prompt diff"
                        sh "echo =============================================="
                        }
                        sh "aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin 374053208103.dkr.ecr.us-west-2.amazonaws.com"
                        sh "docker build -f test.dockerfile -t \$(git rev-parse HEAD) ."
                        sh "docker run --network host --memory=8G --cpus=3 \$(git rev-parse HEAD)"
                    }
                }
            }
        }

    }
}
