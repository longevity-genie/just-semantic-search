#!/bin/bash
script_dir=$(dirname "$(readlink -f "$0")")
docker build -f ${script_dir}/Dockerfile -t ghcr.io/longevity-genie/just-semantic-search/rag-server:local ${script_dir}/../../../
podman pull docker-daemon:ghcr.io/longevity-genie/just-semantic-search/rag-server:local
