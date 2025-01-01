#!/bin/bash

# Partial model name to search for
MODEL_NAME="$1"

# Endpoint URL
URL="http://localhost:11434/api/tags"

# Get the list of models from the endpoint
response=$(curl -s "$URL")

# Check if the response contains the partial model name
if echo "$response" | grep -q "\"name\":\"[^\"]*$MODEL_NAME[^\"]*\""; then
  echo "A model containing '$MODEL_NAME' exists in the local system."
else
  echo "No model containing '$MODEL_NAME' exists in the local system."
fi
