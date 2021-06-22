<h1> Genre_classifier </h1>
<h3> Input Data: </h3>
<p> Data used was collected from GTZAN dataset on kaggle </p>
<h3> Data Preprocessing: </h3>
<p> The 30 second files were split into 3 second files so as to obtain more data <br>
  Obtaining spectrograms and training both pre-trained cnn models and custom made models <br>
  gave high accuracies but very less precision <br>
  So a few features like MFCC were extracted and stored in a dataframe <br>
  The extracted features were exported to a csv file "features.csv" </p>
 <h3> Building the Classifier Model </h3>
  <p> Used a CNN with 3 Conv1D layers and 2 Dense layers <br>
      Performed Stratified K fold to obtain the best model </p>
  <h3> The Flask App </h3>
  <p> The user has to upload a file of .wav extension <br>
  On uploading the file, the user is redirected to another page which shows the predicted result <br>
  The user can click the "PREDICT AGAIN" button to re-preform the prediction </p>
