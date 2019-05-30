#!/bin/bash
bucket=$1
path=$2
echo "Uploading doc to s3://${bucket}/${path}/"
aws s3 sync --delete docs/_build/html/ s3://${bucket}/${path}/ --acl public-read
echo "Uploaded doc to http://${bucket}.s3-accelerate.dualstack.amazonaws.com/${path}/index.html"
