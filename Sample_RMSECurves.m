function [RMSE] = Sample_RMSECurves(Sh1, Sh2, Sh3, sampleTrial)
%%%%%
% Method 1: Cubic Spline
%%%%%

% Step 1 - Create unit vector 
unitVector = zeros(length(sampleTrial), 3);
for a = 1:length(sampleTrial)
    unitVector(a, :) = Sh2(a, :)/norm(Sh2(a, :)); 
end 
    
% Step 2 - Create position vector 
for a = 1:length(sampleTrial)
    positionVector(a, :) = sqrt((Sh2(:, 1).^2) + (Sh2(:, 2).^2) + (Sh2(:, 3).^2));
end
posVec = [positionVector(1, :)]';

% Step 3 - Input missing data length
window = input('How many frames of missing data would you like to experiment with?: ');
% NOTE: the sample trial is from 25 frames prior to heel strike, to 25
% frames following the next heelstrike.  Therefore missing frames <26 is 
% ideal for the sample code in order to evaluate the resulting RMSE curves 
% normalized to gait cycle. 

% Step 4 - Loop through moving window
nan = repmat(NaN, [window, 1]); %filler of NaN's for the missing data
warning('off', 'all')
warning
set(figure('Name', 'Marker Reconstruction', 'NumberTitle', 'off'), 'Units', 'Normalized', 'OuterPosition', [0 0 1 1]);
for b = 1:length(posVec - window); 
    x = posVec';
    x(b + 1:b + window) = nan; %fill missing data with NaN's
    data = x';
    data(isnan(data)) = interp1(find(~isnan(data)), data(~isnan(data)), find(isnan(data)),'cubic'); %find where data isnan; fill with cubic spline interpolation
    all(b, :).all = data;
    check = isequal(size(data), size(posVec));
    if check == 0 %a check to make sure the moving window does not exceed maximum dimensions
        break
    end 
    diff = posVec - data;
    RMS = sqrt(sum(diff.^2)/length(diff));    
    rmse(b,:) = RMS; %collect each iteration of the root mean squared error

    subplot(1,3,1)
    plot(posVec, 'k')
    title('Original Marker Position Vector')
    subplot(1,3,2)
    plot(data, 'r')
    title('Reconstructed Marker Position Vector')
    subplot(1,3,3)
    plot(rmse, '-.r')
    title('Time-varying RMSE')
    pause(0.075)
    hold off
end

% Step 5 - Normalize RMSE curve from heelstrike - heelstrike
last = length(sampleTrial);
last2 = 25 - window;
lFrame = last - window;
if window <= 24 
    rmse1 = rmse(25:lFrame - last2, :); 
else
    rmse1 = rmse(25:lFrame - 1, :); %cut down trial to desired points
end

up = (length(rmse1))/1000; %sample to 1000 points 
X1 = 1:length(rmse1);
Xi = 0:up:length(rmse1);
RMSE.Spline = interp1(X1, rmse1, Xi, 'cubic'); 

%%%%%
% Method 2: Machine Learning
%%%%%

orig = [Sh1 Sh3 Sh2]; %move second marker on the shank cluster to the far right (easier to work with)
nan = repmat(NaN, [window, 3]); 
set(figure('Name', 'Machine Learning Reconstruction', 'NumberTitle', 'off'), 'Units', 'Normalized', 'OuterPosition', [0 0 1 1]);
for c = 1:length(orig - window);
    mach = [Sh1 Sh3 Sh2];
    
    % Step 1 - Input NaN's to replace missings 
    mach(c + 1:c + window, 7:9) = nan; 
    train = []; 
    test = [];
    aa = isnan(mach); 
    
    % Step 2 - Dichotomize matrix into training and testing sets
    for d = 1:length(mach) 
        if any(aa(d, :)) == 1
            test(d, :) = mach(d, :);
        else
            train(d, :) = mach(d, :);
        end
    end 
    train(any(train==0, 2), :) = []; 
    test(any(test==0, 2), :) = []; 
    [rtest, ~] = size(test);
    [rnan, ~] = size(nan);
    check = isequal(rtest, rnan); %a check to make sure moving window does not exceed maximum dimensions
    if check == 0
        break
    end
    
    % Step 3 - Use training set to gather regression coefficients for predictor matrix X, predicting response Y
    pred = train(:, 1:6); 
    YM3z = train(:, 9);
    YM3y = train(:, 8);
    YM3x = train(:, 7);
    [coeffM3z, ~, ~, ~] = regress(YM3z, pred);
    [coeffM3y, ~, ~, ~] = regress(YM3y, pred);
    [coeffM3x, ~, ~, ~] = regress(YM3x, pred);
    predictM3z = [];
    predictM3y = [];
    predictM3x = [];
    for e = 1:length(train)
        predictM3z(e, :) = (pred(e, 1) * coeffM3z(1, 1)) + (pred(e, 2) * coeffM3z(2, 1)) + (pred(e, 3) * coeffM3z(3, 1)) + (pred(e, 4) * coeffM3z(4, 1)) + ...
            (pred(e, 5) * coeffM3z(5, 1)) + (pred(e, 6) * coeffM3z(6, 1)); 
        predictM3y(e, :) = (pred(e, 1) * coeffM3y(1, 1)) + (pred(e, 2) * coeffM3y(2, 1)) + (pred(e, 3) * coeffM3y(3, 1)) + (pred(e, 4) * coeffM3y(4, 1)) + ...
            (pred(e,5)*coeffM3y(5,1)) + (pred(e,6)*coeffM3y(6,1));
        predictM3x(e, :) = (pred(e, 1) * coeffM3x(1, 1)) + (pred(e, 2) * coeffM3x(2, 1)) + (pred(e, 3) * coeffM3x(3, 1)) + (pred(e, 4) * coeffM3x(4, 1)) + ...
            (pred(e, 5) * coeffM3x(5, 1)) + (pred(e, 6) * coeffM3x(6, 1));
    end
    obsM3z = train(:, 9);
    obsM3y = train(:, 8);
    obsM3x = train(:, 7);

    % Step 4 - Predict missing data 
    predtM3z = [];
    predtM3y = [];
    predtM3x = [];
    for f = 1:rtest
        predtM3z(f, :) = (test(f, 1) * coeffM3z(1, 1)) + (test(f, 2) * coeffM3z(2, 1)) + (test(f, 3) * coeffM3z(3, 1)) + (test(f, 4) * coeffM3z(4, 1)) + ...
            (test(f, 5)*coeffM3z(5, 1)) + (test(f, 6) * coeffM3z(6, 1));
        predtM3y(f, :) = (test(f, 1) * coeffM3y(1, 1)) + (test(f, 2) * coeffM3y(2, 1)) + (test(f, 3) * coeffM3y(3, 1)) + (test(f, 4) * coeffM3y(4, 1)) + ...
            (test(f, 5)*coeffM3y(5,1)) + (test(f,6)*coeffM3y(6,1));
        predtM3x(f, :) = (test(f, 1) * coeffM3x(1, 1)) + (test(f, 2) * coeffM3x(2, 1)) + (test(f, 3) * coeffM3x(3, 1)) + (test(f, 4) * coeffM3x(4, 1)) + ...
            (test(f, 5)*coeffM3x(5, 1)) + (test(f, 6)*coeffM3x(6, 1));
    end
    comb = [predtM3x predtM3y predtM3z];
    
    % Step 5 - Recombine data and evaluate RMSE
    mach(c + 1:c + window, 7:9) = comb;
    recon = mach(:, 7:9); 
    orig2 = orig(:, 7:9);
    for g = 1:length(recon)
        reconML(g, :) = sqrt((recon(:, 1).^2) + (recon(:, 2).^2) + (recon(:, 3).^2)); %position vectors
        origML(g, :) = sqrt((orig2(:, 1).^2) + (orig2(:, 2).^2) + (orig2(:, 3).^2));
    end
    diffM = origML - reconML;
    RMSM = sqrt(sum(diffM(1, :).^2)/length(diffM(1, :))); %rmse  
    rmseM(c,:) = RMSM;
    
    %%%%%
    % Method 3: Machine Learning with Circular Solution Space
    %%%%%
    
    % Step 1 - Calculate radius
    Sh12 = (Sh2 - Sh1);
    Sh13 = (Sh3 - Sh1);
    Sh32 = (Sh2 - Sh3);
    pointC = ((Sh1 + Sh3)/2); %midpoint
    for h = 1:length(pointC)
        radi(h, :) = sqrt((Sh2(h, 1) - pointC(h, 1)).^2 + ((Sh2(h, 2) - pointC(h, 2)).^2 + ((Sh2(h, 3) - pointC(h, 3)).^2)));
    end
    radiu = mean(radi);
    
    % Step 2 - Unit vector calculation for the shank cluster
    clShi = zeros(length(Sh2), 3); 
    clShTj = zeros(length(Sh2), 3);
    clShk = zeros(length(Sh2), 3);
    clShj = zeros(length(Sh2), 3);
    arbPoint = [randi(500, length(Sh2), 1) randi(500, length(Sh2), 1) randi(500, length(Sh2), 1)]; %random point in space to aid in cluster coordinate system calculation
    for j = 1:length(Sh2)
        clShi(j, :) = (Sh1(j, :) - pointC(j, :))/norm(Sh1(j, :) - pointC(j, :)); 
        clShTj(j, :) = cross(arbPoint(j, :), Sh1(j, :))/norm(cross(arbPoint(j, :), Sh1(j, :)));
        clShk(j, :) = cross(clShi(j, :), clShTj(j, :))/norm(cross(clShi(j, :), clShTj(j, :)));
        clShj(j, :) = cross(clShk(j, :), clShi(j, :))/norm(cross(clShk(j, :), clShi(j, :)));
    end 
    
    vecA = clShj;
    vecB = clShk;
    circ = degtorad(linspace(0, 360, 1000));

    % Step 3 - Equation of a circle
    xTheta = [];
    yTheta = [];
    zTheta = [];
    for k = 1:length(pointC)
        for j = 1:length(circ)
            xTheta(j, k) = (pointC(k, 1) + (radiu*cos(circ(j))*vecA(k, 1)) + (radiu*sin(circ(j))*vecB(k, 1)));
            yTheta(j, k) = (pointC(k, 2) + (radiu*cos(circ(j))*vecA(k, 2)) + (radiu*sin(circ(j))*vecB(k, 2)));
            zTheta(j, k) = (pointC(k, 3) + (radiu*cos(circ(j))*vecA(k, 3)) + (radiu*sin(circ(j))*vecB(k, 3)));
        end
    end 
    
    % Step 4 - Find point on the circle for each frame that closely
    % corresponds to Machine Learning (i.e., minimum Euclidean distance)
    xTheta1 = [xTheta(:, c + 1:c + window)];
    yTheta1 = [yTheta(:, c + 1:c + window)];
    zTheta1 = [zTheta(:, c + 1:c + window)]; 
    dex = zeros(length(comb), 1);
    combCirc = zeros(length(comb), 3); 
    for m = 1:length(comb)
        comTheta = [xTheta1(:, m) yTheta1(:, m) zTheta1(:, m)]; %get equations for the circle at the frames specified by the moving window
        n = size(comb(m, :), 1); %Lines 244 - 249 adapted from code written by Piotr Dollar (2009) from pdist2 function, available at www.mathworks.com
        o = size(comTheta, 1);
        oT = comTheta';
        for p = 1:o
            EucDist(:, p) = sqrt((comb(m, 1) - oT(1, p)).^2 + (comb(m, 2) - oT(2, p)).^2 + (comb(m, 3) - oT(3, p)).^2);
        end
        distMin = EucDist(1, :)';
        index = find(distMin == min(distMin)); %index to find minimum Euclidean distance
        dex(m, :) = index(1, 1);
        combCirc(m, :) = comTheta(index(1, 1), :);
    end
    ndex(c, :).all = dex; %save data for each window
    distall(c, :).all = combCirc;
    
    % Step 5 - Recombine data
    mach2 = mach;
    mach2(c + 1:c + window, 7:9) = combCirc; 
    recon3 = mach2(:, 7:9);
    for q = 1:length(recon3)
        reconML2(q, :) = sqrt((recon3(:, 1).^2) + (recon3(:, 2).^2) + (recon3(:, 3).^2));
    end
    diffM2 = origML - reconML2;
    RMSM2 = sqrt(sum(diffM2(1, :).^2)/length(diffM2(1, :)));
    rmseM2(c, :) = RMSM2;
    
    subplot(2,3,1)
    plot(origML(1, :), 'k')
    title('Original Marker Position Vector')
    subplot(2,3,2)
    plot(reconML(1, :), 'r')
    title('Machine Learning Position Vector')
    subplot(2,3,3)
    plot(rmseM, '-.r')
    title('Time-varying RMSE')
    subplot(2,3,4)
    plot(origML(1, :), 'k')
    title('Original Marker Position Vector')
    subplot(2,3,5)
    plot(reconML2(1, :), 'r')
    title('Machine Learning + CSS Position Vector')
    subplot(2,3,6)
    plot(rmseM2, '-.r')
    title('Time-varying RMSE')
    pause(0.075)
    hold off 
end

% Step 6 - Normalize RMSE curves to gait cycle 

if window <= 24 
    rmseML = rmseM(25:lFrame - last2, :); 
else
    rmseML = rmseM(25:lFrame - 1, :); 
end
up2 = (length(rmseML))/1000; 
X12 = 1:length(rmseML);
Xi2 = 0:up2:length(rmseML);
RMSE.MachLearn = interp1(X12, rmseML, Xi2, 'cubic');

if window <= 24 
    rmseML2 = rmseM2(25:lFrame - last2, :); 
else
    rmseML2 = rmseM2(25:lFrame - 1, :); 
end
up3 = (length(rmseML2))/1000; 
X13 = 1:length(rmseML2);
Xi3 = 0:up3:length(rmseML2);
RMSE.MachLearnCSS = interp1(X13, rmseML2, Xi3, 'cubic');

end