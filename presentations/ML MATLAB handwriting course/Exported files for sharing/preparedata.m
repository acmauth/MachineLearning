%% 
% Make a datastore to the raw data

letterds = datastore("Data\*.txt");
%% 
% Make transformed datastores by adding preprocessing and feature-extraction 
% functions

procletds = transform(letterds,@preprocess);
letfeatds = transform(procletds,@extractfeatures);
%% 
% Get the features for all the data

letterdata = readall(letfeatds);
%% 
% Add a variable containing the known letter category (obtained from the 
% file name)

letterdata.Character = categorical(extractBetween(letterds.Files,"_","_"))
%% 
% Divide data into training and testing sets

rng(123)  % Set random number state, for reproducibility
% Create a holdout partition and get indices for training and testing
part = cvpartition(height(letterdata),"HoldOut",0.25)
idxtest = test(part);
idxtrain = training(part);
% Index into data to get training and testing sets
testdata = letterdata(idxtest,:);
traindata = letterdata(idxtrain,:);
%% 
% Keep a record of the raw data files corresponding to the training and 
% testing data

files = extractAfter(letterds.Files,"Data"+filesep)
testfiles = files(idxtest);
trainfiles = files(idxtrain);
%% 
% Check class distribution of resulting data sets

histogram(testdata.Character)
histogram(traindata.Character)
%% 
% Save results

save letterfeatures testdata traindata testfiles trainfiles
%% 
% Helper functions for preprocessing data and extracting features

function data = preprocess(data)
% Normalize time [0 1]
data.Time = (data.Time - data.Time(1))/(data.Time(end) - data.Time(1));
% Fix aspect ratio
data.X = 1.5*data.X;
% Center X & Y at (0,0)
data.X = data.X - mean(data.X,"omitnan");
data.Y = data.Y - mean(data.Y,"omitnan");
% Scale to have bounding box area = 1
scl = 1/sqrt(range(data.X)*range(data.Y));
data.X = scl*data.X;
data.Y = scl*data.Y;
% Calculate derivatives
data.dXdT = [NaN;diff(data.X)./diff(data.Time)];
data.dYdT = [NaN;diff(data.Y)./diff(data.Time)];
data.dXdT(isinf(data.dXdT)) = NaN;      % Infinite value => same values in data.Time
data.dYdT(isinf(data.dYdT)) = NaN;      % => derivative calculation is meaningless
% Smooth derivatives using a moving average
n = round(0.1*numel(data.Time));
data.dXdT = movmean(data.dXdT,n);
data.dYdT = movmean(data.dYdT,n);
end

function feat = extractfeatures(letter)
% Correlations between signals (not including time)
c = corrcoef(letter{:,2:end},"Rows","pairwise");
% Set some parameters for finding local mins/maxes
mp = 0.1;
derivscale = 2;
% Calculate features
feat = [range(letter.Y)/range(letter.X),...                             % aspect ratio
    mad(letter.X), mad(letter.Y),...                                    % deviation for X & Y
    mean(letter.dXdT,"omitnan"),mad(letter.dXdT),...                    % mean and deviation
    mean(letter.dYdT,"omitnan"),mad(letter.dYdT),...                    %   for X' & Y'
    c(1,2:end),c(2,3:end),c(3,4:end),c(4,end),...                       % correlations
    nnz(islocalmin(letter.X,"MinProminence",mp)),...                    % number of local
    nnz(islocalmax(letter.X,"MinProminence",mp)),...                    %   min/max for X & Y
    nnz(islocalmin(letter.Y,"MinProminence",mp)),...
    nnz(islocalmax(letter.Y,"MinProminence",mp)),...
    nnz(islocalmin(letter.dXdT,"MinProminence",derivscale*mp)),...      % number of local
    nnz(islocalmax(letter.dXdT,"MinProminence",derivscale*mp)),...      %   min/max for X' & Y'
    nnz(islocalmin(letter.dYdT,"MinProminence",derivscale*mp)),...
    nnz(islocalmax(letter.dYdT,"MinProminence",derivscale*mp))];

% Combine features together into a table
feat = array2table(feat,"VariableNames",...
    ["AspectRatio",...
    "MADX","MADY","AvgU","MADU","AvgV","MADV",...
    "CorrXY","CorrXP","CorrXU","CorrXV",...
    "CorrYP","CorrYU","CorrYV","CorrPU","CorrPV","CorrUV",...
    "NumXMin","NumXMax","NumYMin","NumYMax",...
    "NumUMin","NumUMax","NumVMin","NumVMax"]);

end