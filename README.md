<h1> Genre_classifier </h1>
<h3> Input Data: </h3>
<h6> Data used was collected from GTZAN dataset on kaggle </h6>
<h3> Data Preprocessing: </h3>
<h6> The 30 second files were split into 3 second files so as to obtain more data <br>
  Obtaining spectrograms and training both pre-trained cnn models and custom made models <br>
  gave high accuracies but very less precision (<0.5) <br>
  So a few features like MFCC, ZFC were extracted and stored in a dataframe <br>
  The extracted features were exported to a csv file "features.csv" </h6>
 <h3> Building the Classifier Model </h3>
  <h6> Used a CNN with 3 Conv1D layers and 2 Dense layers <br>
      Performed Stratified K fold to obatin the best model </h6>
  <h3> The Flask App </h3>
  <h6> The user has to upload a file of .wav extension <br>
  On uploading the file, the user is redirected to another page which shows the predicted result <br>
  The user can click the "PREDICT AGAIN" button to re-preform the prediction </h6>
