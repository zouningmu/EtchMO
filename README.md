# EtchMO


EtchMO is an open-source library for **differentiable etch simulation** and **etch-aware correction** in computational lithography.  
The code is implemented in **PyTorch** and accelerated with CUDA, making it easy to integrate into gradient-based optimization pipelines.
***
## 1 Method Overview

### 1.1 Differentiable etch simulation

The etch simulator takes a resist image as input and predicts the etched image through the following steps:

- extract boundary responses from the resist image,
- evaluate etch-related kernel responses,
- compute boundary etch bias,
- extend the bias from the boundary to the effective region,
- construct an implicit function,
- perform redistancing to obtain an approximate signed distance field (SDF),
- evolve the SDF using the extended bias,
- recover the final etched pattern.

### 1.2 Pixel-wise EPC

In addition to end-to-end etch-aware optimization, this repo also includes a level-set-based EPC flow.  
Instead of performing conventional segment-wise contour movement, the EPC module updates the retargeted pattern in a **pixel-wise, objective-driven** manner. 

***
## 2 Installation 

### 2.1 Requirements

* Python: 3.7+  
* Pytorch >= 1.8.0



### 2.2 Install
**Prepare Environment**
```bash
conda create -n etchmo python=3.8
conda activate etchmo
```

**Pip Install**
```bash
pip install -r requirements.txt
```
***
## 3 Quick Start
### 3.1 Etch Simulation
Run etch simulation on an input continuous resist image:
```bash
python src/run_etch.py \
  --input samples/test_32nm.png \
  --out outputs/etch_demo \
  --H 2048 --W 2048
```
### 3.2 EPC
Run EPC on a single ```.gds``` testcase:
```bash
python src/epc.py \
  --gds samples/test_45nm.gds \
  --out outputs/epc_demo 
```