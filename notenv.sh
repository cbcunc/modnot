#! /bin/bash

# Prepare a vanilla Anaconda 3 environment for the model notebook

echo ==========================
echo Adding conda-forge channel
echo ==========================
echo
conda config --add channels conda-forge

echo ===================
echo Updating ipywidgets
echo ===================
echo
conda update ipywidgets

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
jupyter nbextension enable --py widgetsnbextension

echo ===========================
echo Enabling Jupyter Dashboards
echo ===========================
echo
pip install jupyter_dashboards

echo =============================
echo Setting up Jupyter Dashboards
echo =============================
echo
jupyter dashboards quick-setup --sys-prefix

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
