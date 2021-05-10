# AdvancedPython2021

This repo hosts our codes for Advanced Python for Data Science (Spring 2021). Our goals are two-folded. First, to detect the human faces within a video and apply filter effects on them. Second, we would like to accelerate this process as much as possible using techniques including:

- Cython
- Array-wise operation
- Numba
- Multiprocessing
- CUDA Programming

The repo is structured as the following:
- All of our solutions are organized in ```final_submission.ipynb``` with detailed documentation. We developed our solution in Google Colab.
- The ```data``` folder contains videos for testing and generated from our algorithms.
- The ```results``` folder contains the results of our experiments comparing our solutions.

For our face detection task, we acknowledge two GitHub projects:
- Tim Esler's implementation of MTCNN in PyTorch: https://github.com/timesler/facenet-pytorch
- Adam Geitgey 's implementation of several trained face detection models, which are often more efficient for CPUs: https://github.com/ageitgey/face_recognition
