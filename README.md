# Guitar Chord AI

AI system for extracting guitar chord images from fretboards.

Supported chords:
C, G, Am, D, Em, F

Pipeline:
1. Collect dataset from the internet
2. Clean images
3. Train CNN (EfficientNet)
4. Evaluate models
5. Output
6. Grad-CAM visualization

Run:
1. Install dependencies
pip install -r requirements.txt

2. Download the dataset
python src/dataset.py

3. Clean the dataset
python src/preprocessing.py

4. Train the model
source python/train.py

5. Evaluate the model
python src/evaluate.py