% This script is for obtaining the statistical results after training is
% completed in the "CrackSegmentation.m" script. 

% get the predicted results
pxdsResults = semanticseg(testx,net,"WriteLocation",tempdir, 'MiniBatchSize',10);

% get global statistical results
metrics = evaluateSemanticSegmentation(pxdsResults,testy);
metrics.ClassMetrics

% get per-class statistical results
evaluationMetrics = ["accuracy" "bfscore" "iou"];
metrics = evaluateSemanticSegmentation(pxdsResults,testy,"Metrics",evaluationMetrics);
metrics.ClassMetrics
