#!/bin/bash
set -x
bumpversion --current-version $(git tag | grep -E '^[0-9]+\.[0-9]+\.[0-9]+' | sort -V | tail -n 1) patch --tag --tag-name={new_version}
NEW_VERSION=$(git tag | grep -E '^[0-9]+\.[0-9]+\.[0-9]+' | sort -V | tail -n 1)
git push origin $NEW_VERSION
echo $NEW_VERSION