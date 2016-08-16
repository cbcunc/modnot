#! /bin/bash

# Copyright (c) 2016, University of North Carolina Renaissance Computing Institute

# Prepare a vanilla Anaconda 3 environment for the model notebook

echo ===================
echo Updating ipywidgets
echo ===================
echo
conda update -y ipywidgets

echo =======================
echo Reinstalling matplotlib
echo =======================
echo
conda install -y -c conda-forge/label/rc -c conda-forge matplotlib

echo ===================
echo Reinstalling libpng
echo ===================
echo
conda install -y -c conda-forge/label/rc -c conda-forge libpng

echo =================
echo Installing ipympl
echo =================
echo
pip install ipympl

echo ===============
echo Enabling ipympl
echo ===============
echo
jupyter nbextension install --py --symlink --user ipympl
jupyter nbextension enable --py --user ipympl
jupyter nbextension enable --py --sys-prefix --user widgetsnbextension

echo ===========================
echo Installing Jupyter Dashboards
echo ===========================
echo
pip install jupyter_dashboards

echo =============================
echo Setting up Jupyter Dashboards
echo =============================
echo
jupyter dashboards quick-setup --sys-prefix
jupyter nbextension enable jupyter_dashboards --py --sys-prefix

echo ==================
echo Installing basemap
echo ==================
echo
conda install -y -c conda-forge/label/rc -c conda-forge basemap

echo =============================
echo Installing basemap hires data
echo =============================
echo
conda install -y -c conda-forge/label/rc -c conda-forge basemap-data-hires

echo =================
echo Installing owslib
echo =================
echo
conda install -y -c conda-forge/label/rc -c conda-forge owslib
