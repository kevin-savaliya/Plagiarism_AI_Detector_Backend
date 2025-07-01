#!/bin/bash
apt-get update
apt-get install -y libpython3.11-dev build-essential libblas-dev liblapack-dev gfortran
rm -rf /var/lib/apt/lists/*
