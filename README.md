# FaceLean
An experiment to use a webcam as a game input device. More detail on this project is available on my blog: [https://www.gdcorner.com/2020/03/31/WebcamAsGameInput.html](https://www.gdcorner.com/2020/03/31/WebcamAsGameInput.html)

## Main tech used

* [Python](https://www.python.org/)
* [NumPy](https://numpy.org/)
* [OpenCV](https://opencv.org/)
* [DLib](http://dlib.net/)

### Requirements & installation

Ensure you have the following software installed

* CMake (latest)
* Visual Studio Community or Professional (I used 2019)
* Python 3.7 or greater

Download or clone the repo.

Run the following command to install other requirements

```bash
pip install -r requirements.txt
```

Download and extract shape_predictor_5_face_landmarks.dat from [https://github.com/davisking/dlib-models/blob/master/shape_predictor_5_face_landmarks.dat.bz2](https://github.com/davisking/dlib-models/blob/master/shape_predictor_5_face_landmarks.dat.bz2) into the project directory

Then run main.py

```bash
python main.py
```
