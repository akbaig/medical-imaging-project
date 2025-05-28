# Medical Imaging Project

## Project Overview

This project involves handling DICOM medical images, where the goal is to load, visualize, and perform coregistration of these images. Using Python and various libraries, the project tackles tasks from basic DICOM operations to more complex image processing techniques.

## Objectives

- **DICOM Loading and Visualization**
  - Load and visualize DICOM images from the RadCTTACEomics dataset.
  - Manage and rearrange image data based on specific DICOM headers.
  - Create animations to visualize Maximum Intensity Projections of the imaging data.

- **Semi-automatic Segmentation of Tumor**
  - Implements Region-Growing Technique seeded at or around the tumor centroid

## Installation

To set up the project environment, follow these steps:

```bash
git clone https://github.com/akbaig/medical-imaging-project
cd medical-imaging-project
pip install -r requirements.txt
```

## Usage

- Run the files `task1.py` and `task2.py` to execute the code.

## Contributions

This project is part of a course requirement for Medical Image Processing at UIB, Spain. Contributions are welcomed once repository is public.

## Acknowledgments

Special thanks  to [Prof. Pedro Bibiloni Serrano](https://www.uib.es/es/personal/ABjI3MDA5NA/) for providing the opportunity to work on this project.