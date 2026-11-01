# Meshroom + AliceVision Setup (Linux)

Meshroom is an open-source photogrammetry pipeline built on AliceVision. This guide installs AliceVision 3.3.0 and the matching Meshroom commit.

## 1. Install AliceVision

Download and extract the pre-built release:

```bash
curl -L -O https://github.com/alicevision/AliceVision/releases/download/v3.3.0/AliceVision-3.3.0-Linux.tar.gz
mkdir -p alicevision
tar -xvf AliceVision-3.3.0-Linux.tar.gz -C ./alicevision/ --strip-components=1
```

## 2. Clone Meshroom

Check out the commit compatible with AliceVision 3.3.0:

```bash
git clone git@github.com:alicevision/Meshroom.git
cd Meshroom
git checkout 024b6f398c67bec4968a29a2c5744e49e2bab9b8
```

## 3. Set environment variables

Append the following to `~/.bashrc`, replacing the two placeholder paths:

```bash
# AliceVision / Meshroom
export ALICEVISION_ROOT=<path-to-alicevision>
export MESHROOM_ROOT=<path-to-meshroom>

export PATH="$ALICEVISION_ROOT/bin:$ALICEVISION_ROOT/lib:$PATH"
export LD_LIBRARY_PATH="$ALICEVISION_ROOT/bin:$ALICEVISION_ROOT/lib:$LD_LIBRARY_PATH"

export MESHROOM_NODES_PATH="$ALICEVISION_ROOT/share/meshroom"
export MESHROOM_PIPELINE_TEMPLATES_PATH="$ALICEVISION_ROOT/share/meshroom"

export ALICEVISION_SENSOR_DB="$ALICEVISION_ROOT/share/aliceVision/cameraSensors.db"
export ALICEVISION_VOCTREE="$ALICEVISION_ROOT/share/aliceVision/vlfeat_K80L3.SIFT.tree"
export ALICEVISION_SPHERE_DETECTION_MODEL="$ALICEVISION_ROOT/share/aliceVision/sphereDetection_Mask-RCNN.onnx"
export ALICEVISION_SEMANTIC_SEGMENTATION_MODEL="$ALICEVISION_ROOT/share/aliceVision/fcn_resnet50.onnx"
export ALICEVISION_COLORCHARTDETECTION_MODEL_FOLDER="$ALICEVISION_ROOT/share/aliceVision/ColorChartDetectionModel"
export ALICEVISION_LENS_PROFILE_INFO=""

export ALICEVISION_USE_OPENCV=ON
export ALICEVISION_USE_POPSIFT=ON
export ALICEVISION_USE_CCTAG=ON
export ALICEVISION_BUILD_SWIG_BINDING=ON
export ALICEVISION_INSTALL_MESHROOM_PLUGIN=ON
```

Then reload:

```bash
source ~/.bashrc
```

## 4. Install Meshroom dependencies

```bash
cd "$MESHROOM_ROOT"
conda create -n meshroom python=3.10
conda activate meshroom
pip install -r requirements.txt -r dev_requirements.txt
pip install pycolmap==4.1.1
```

## 5. Verify

Run a batch reconstruction to confirm everything is wired up:

```bash
python "$MESHROOM_ROOT/bin/meshroom_batch" \
    -i <path-to-input-images> \
    -o <path-to-output-folder> \
    -p photogrammetry \
    -s <path-to-save-file>
```