# Docker
 For model training and better inference purposes use docker containers.
 
Latest version of the container is `v0.1`.

## Building image
To build latest docker image:

```commandline
bash build
```

This command will create image with `water_drops_processing:{YOUR_NAME}-v0.1` tag. 
Alternatively you can load built image from *.tar archive.

## Running container
To run the latest version of the facade_processing_tools container:

```commandline
bash run_single_gpu CONT_NAME DEVICE_ID MNT_DIR
```
Where 
- `CONT_NAME` is the name of the container.
- `DEVICE_ID` is the id of the selected GPU, use `all` to mount all GPU's.
- `MNT_DIR` path to the mounted folder to the `/mnt` location in the container.
