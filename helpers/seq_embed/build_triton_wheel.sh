#!/bin/bash
# Build a wheel compatible with DNABERT.
#   DNABERT optionally uses code for flash_attention which depends on the legacy implementation of Triton
#   This subsequently affects DNABERT-S.
# It is possible to run without triton, in which case a different slower implementation is used by DNABERT

wget https://github.com/triton-lang/triton/archive/refs/tags/legacy-backend.tar.gz && \
  tar xzvf legacy-backend.tar.gz && \
  cd triton-legacy-backend/python && \
  python setup.py bdist_wheel
