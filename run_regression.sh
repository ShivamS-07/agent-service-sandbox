#!/bin/bash
VERSION_TAG=${1:-$(git rev-parse HEAD)}
echo "Kicking off regression test for version $VERSION_TAG"
aws sqs send-message --queue-url https://sqs.us-west-2.amazonaws.com/374053208103/insights-backend-dev-boosted-dag --message-body \
"{\"AgentServiceRegressionTest\": {}, \"service_versions\": {\"agent_service_version\": \"$VERSION_TAG\"}}"