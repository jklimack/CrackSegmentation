clear all;


% DATA
% =========================================
% paths to data
path_train_data = "DeepCrack/train_img";
path_train_labels = "DeepCrack/train_lab";
path_test_data = "DeepCrack/test_img";
path_test_labels = "DeepCrack/test_lab";

% read image data
trainx = imageDatastore(path_train_data);
testx = imageDatastore(path_test_data);

% read label data
classNames = ["crack", "concrete"];
pixelLabelID = [255 0];
trainy = pixelLabelDatastore(path_train_labels,classNames,pixelLabelID);
testy = pixelLabelDatastore(path_test_labels,classNames,pixelLabelID);

% class weighting
tbl = countEachLabel(trainy)
imageFreq = tbl.PixelCount ./ tbl.ImagePixelCount
classWeights = median(imageFreq) ./ imageFreq
classWeights(1) = classWeights(1)*1.1;
classWeights;


% BUILD NETWORK
% =========================================
num_filters = [64, 64, 64, 64, 64];
filterSize = 3;

layers = [
    imageInputLayer([384 544 3])
    convolution2dLayer(filterSize,num_filters(1),'Padding',1, 'Name', 'conv1')
    reluLayer()
    maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer(filterSize,num_filters(2),'Padding',1, 'Name', 'conv2')
    reluLayer()
    %maxPooling2dLayer(2,'Stride',2)
    %convolution2dLayer(filterSize,num_filters(3),'Padding',1, 'Name', 'conv3')
    %reluLayer()
    %maxPooling2dLayer(2,'Stride',2)
    %convolution2dLayer(filterSize,num_filters(4),'Padding',1, 'Name', 'conv4')
    %reluLayer()
    %maxPooling2dLayer(2,'Stride',2)
    %convolution2dLayer(filterSize,num_filters(5),'Padding',1, 'Name', 'conv5')
    %reluLayer()
    
    %transposedConv2dLayer(4,num_filters(5),'Stride',2,'Cropping',1);
    %transposedConv2dLayer(4,num_filters(4),'Stride',2,'Cropping',1);
    %transposedConv2dLayer(4,num_filters(3),'Stride',2,'Cropping',1);
    transposedConv2dLayer(4,num_filters(2),'Stride',2,'Cropping',1);
    
    convolution2dLayer(1,2);
    softmaxLayer()
    pixelClassificationLayer('Name','labels','Classes',tbl.Name,'ClassWeights',classWeights)
]

% TRAIN NETWORK
% =========================================

trainData = pixelLabelImageDatastore(trainx, trainy);
% [trainData, valData] = splitEachLabel(trainData, 0.9);

opts = trainingOptions('sgdm', ...
                'InitialLearnRate',1e-3, ...
                'MaxEpochs',25, ...
                'MiniBatchSize',2, ...
                'Plots','training-progress', ...
                'Shuffle', 'every-epoch');

net = trainNetwork(trainData,layers,opts);


