%% 
% Load processed data from file.

load letterfeatures
%% Fit, predict, and evaluate
% Try a simple kNN.

kNNmodel = fitcknn(traindata,"Character")
%% 
% Evaluate the model on the test data.

loss(kNNmodel,testdata)
predLetter = predict(kNNmodel,testdata);
confusionchart(testdata.Character,predLetter);
%% 
% Try more neighbors. Because kNN is distance-based, add normalization.

kNNmodel = fitcknn(traindata,"Character","NumNeighbors",10,"Standardize",true)
loss(kNNmodel,testdata)
%% 
% Try a different number of neighbors but add weighting by distance.

kNNmodel = fitcknn(traindata,"Character","NumNeighbors",5,"Standardize",true, ...
    "DistanceWeight","squaredinverse");
loss(kNNmodel,testdata)
%% 
% Try a different approach: an ensemble of bagged trees.

rng(123)
TBmodel = fitcensemble(traindata,"Character","Method","Bag","Learners","tree","NumLearningCycles",30);
loss(TBmodel,testdata)
%% 
% Looks like that's about as good as it's going to get. Go back to the weighted 
% kNN model with k = 5.

predLetter = predict(kNNmodel,testdata);
confusionchart(testdata.Character,predLetter);
%% Analyze misclassifications
% Get the confusion matrix data.

cm = confusionmat(testdata.Character,predLetter);
% Split into correct and incorrect classifications
yes = diag(cm);
no = cm - diag(yes);
%% 
% Get misclassification rate for each letter.

misratebyletter = sum(no,2)./sum(cm,2);
%% 
% Convert to a table with letter names and misclassification rate.

letters = categories(traindata.Character);
misratebyletter = table(letters,misratebyletter,'VariableNames',["Letter","MisClassRate"]);
% Sort by worst misclassification
misratebyletter = sortrows(misratebyletter,"MisClassRate","descend")
bar(misratebyletter.MisClassRate)
xticks(1:26)
xticklabels(misratebyletter.Letter)
%% 
% Look at individual misclassification examples.

letter = "U";
%% 
% Get the observations with this true class that were misclassified as something 
% else.

misclassidx = (testdata.Character == letter) & (predLetter ~= testdata.Character);
%% 
% Make a table of the misclassified observations, with the predicted letter 
% included.

badpred = testdata(misclassidx,:);
badpred.Prediction = predLetter(misclassidx);
%% 
% Get the associated data files. Read the raw data from each file and plot 
% it. Label the plot with the prediction.

badfiles = testfiles(misclassidx);
for k = 1:numel(badfiles)
    badletter = readtable("Data"+filesep+badfiles(k));
    figure
    plot(1.5*badletter.X,badletter.Y,".-")
    title("Predicted: "+string(badpred.Prediction(k))+"  --   Actual: "+letter)
    axis equal
end