# FOOL CLASSIFIER

**FOOL CLASSIFIER** is an algorithm to fool a binary classifier named _target-classifier_.

In this regard, we only have access to following information:
1. The _target-classifier_ is a binary classifier classifying data to two categories, _i.e._, **class-1** and **class-0**.
2. We have access to part of classifiers' training data, _i.e._, a sample of 540 paragraphs. 180 for **class-1**, and 360 for **class-0**, provided in the files: _class-1.txt_ and _class-0.txt_ respectively.
3. The _target-classifier_ belong to the SVM family.
4. The _target-classifier_ allows **EXACTLY 20 DISTINCT** modifications in each test sample. **NOTE:** **ADDING** or **DELETING** one word at a time is **ONE** modification. **REPLACING** will be considered as **TWO** modifications (_i.e._, **DELETING** followed by **ADDING**).
5. We are provided with a test sample of 200 paragraphs from **class-1** (in the file: _test_data.txt_). We can use these test samples to get feedback from the _target-classifier_ (only 15 attempts allowed).
6. We are not allowed to use the data _test_data.txt_ for model training.

### Implementation Details
- The function **fool_classifier()** in the file submission.py will be used to read a text file named: _test_data.txt_ from Present Working Directory (PWD), and writes out the modified text file: _modified_data.txt_ in the same directory. Each line in the input file corresponds to a single test sample, so the output file will be written in the same format.
- The implementation of strategy class in the file helper.py will be used for model training and inference building.

### How to Run
python -c "from submission import *; fool_classifier('test_data.txt')"