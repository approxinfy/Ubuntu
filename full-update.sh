#!/bin/bash

if [ "$EUID" -ne 0 ]
  then echo "Error: Please run as root."
  exit
fi

clear

echo "################################################################################"
echo "Updating list of available packages..."
echo "--------------------------------------------------------------------------------"
apt update
echo "################################################################################"
echo

echo "################################################################################"
echo "Upgrading the system by removing/installing/upgrading packages..."
echo "--------------------------------------------------------------------------------"
apt full-upgrade --yes
echo "################################################################################"
echo

echo "################################################################################"
echo "Removing automatically all unused packages..."
echo "--------------------------------------------------------------------------------"
apt autoremove --yes
echo "################################################################################"
echo

echo "################################################################################"
echo "Clearing out the local repository of retrieved package files..."
echo "--------------------------------------------------------------------------------"
apt autoclean --yes
echo "################################################################################"
echo
