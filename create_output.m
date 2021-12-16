% The purpose of this script is to save the resulting segmented images to
% files. 

for i=[1:length(testx.Files)]
    img = readimage(testx, i);
    s = size(img);
    if s(1) == 544
        img = permute(img, [2 1 3]);
    end
    C = semanticseg(img,net);
    CMask = C == "crack";
    [p, n, e] = fileparts(testx.Files{i});
    imwrite(CMask, strcat("output/", n, ".png"));
end

