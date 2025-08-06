# ARX R5 Robot URDF

This directory should contain the URDF file for the ARX R5 robot arm.

## Required Files

- `R5a.urdf`: The URDF description of the ARX R5 robot arm

## Installation

1. Obtain the R5a.urdf file from your ARX R5 robot package or SDK
2. Place it in this directory: `src/lerobot/robots/arx_r5_dual/assets/R5a/R5a.urdf`

## Note

The ARX R5 interface will look for the URDF file in this location. If not found, it will attempt to use a system-wide installation of the URDF file.