# ReTap: ReTune's updrsTapping toolbox

### Summary
This toolbox enables automated accelerometer-based assessment of UPDRS-tapping tasks, performed in clinical Parkinson-assessment routines.
The toolbox will provide 1) an automated prediction of UPDRS-subscore of a 10-second fingertapping task, and 2) provide detailed movement
features which can help the clinician or researcher to assess motor performance and bradykinesia.
<br><br>Written by Jeroen Habets and Rachel Spooner as part of the <a href="https://sfb-retune.de">ReTune-Consortium</a>, a scientific collaboration between (among others) the HHU Düsseldorf and the Charité Berlin.

#### Current state
Work in Progress: data analysis ongoing.

### Requirements:

#### Python:
environment installation:
<c>conda create --name updrsTapping python=3.9 jupyter pandas scipy numpy matplotlib statsmodels seaborn scikit-learn h5py</c>

additional installed packages:
<c>pip install mne</c> (for working with raw .poly5 files, e.g. via TMSI-amplifier)
<c>conda install openpyxl</c> (for opening latest version Excel files)
<c>conda install pingouin</c> (for calculating ICC)


#### Matlab:
