#!/bin/bash

# The tag used will be the hash of the current commit if no argument is provided.
# Otherwise, the tag used will be the argument provided.
VERSION_TAG=${1:-$(git rev-parse HEAD)}

aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin 374053208103.dkr.ecr.us-west-2.amazonaws.com
docker build --platform=linux/amd64 -f test.dockerfile . -t 374053208103.dkr.ecr.us-west-2.amazonaws.com/agent-service-test:$VERSION_TAG
docker build --platform=linux/amd64 -f test.dockerfile . -t 374053208103.dkr.ecr.us-west-2.amazonaws.com/agent-service-test:latest
docker push 374053208103.dkr.ecr.us-west-2.amazonaws.com/agent-service-test:$VERSION_TAG
docker push 374053208103.dkr.ecr.us-west-2.amazonaws.com/agent-service-test:latest
