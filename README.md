# Project Description

Here will be description

## Installation

For repo setup use following methods:
### conda

Automatically setup new conda env named `water_drops_processing` using environment.yml:

```commandline
conda env create --file environment.yml
```

Update existing `water_drops_processing` env:

```commandline
conda activate facade_processing_tools
conda env update --file environment.yml --prune
```

### Docker

Setup docker image with all necessary packages: 

```commandline
cd docker
bash build
bash run_container CONT_NAME DEVICE_ID MNT_DIR
```
Where 
- `CONT_NAME` is the name of the container.
- `DEVICE_ID` is the id of the selected GPU, use `all` to mount all GPU's.
- `MNT_DIR` path to the mounted folder to the `/mnt` location in the container.
