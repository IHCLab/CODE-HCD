close all; clear all

%% Load dataset
dataset= 'Jiangsu';
   
if dataset == 'Jiangsu'

    load ./Dataset/zuixin/river_before.mat
    river_before(:,:,[1:4 49:58 68:74 90:107 139:165 177 196:198])=[];
    x1 = river_before;
    load ./Dataset/zuixin/river_after.mat
    river_after(:,:,[1:4 49:58 68:74 90:107 139:165 177 196:198])=[];
    x2 = river_after;
    clear imgh imghl
    load ./Dataset/zuixin/groundtruth.mat
    gt = lakelabel_v1;
    clear lakelabel_v1
    gt = mat2gray(gt);

    fid = fopen("./config/config.yaml");
    data = textscan(fid, '%s', 'Delimiter', '\n', 'CollectOutput', true);
    fclose(fid);
    data{1}{1}='data_name: river';
    fid = fopen("./config/config.yaml", 'w');
    for I = 1:length(data{1})
        fprintf(fid, '%s\n', char(data{1}{I}));
    end
    fclose(fid);

    Data = "River";

end

%% Initialization

OA_all=[];
kappa_all=[];
precision_all=[];
recall_all=[];
time_all=[];


[x, y, z] = size(x1); 

%% CODE-HCD

for i=1:1

% Deep learning

tic

% activate environment & obtain deep learning output
system(['python main.py']);
time_DL=toc;

load './temp files/Cdl.mat'
load './temp files/index.mat'

cdl = double(reshape(Cdl,y ,x)');

% ADMM

tic
ctemp = CODEsam(x1,x2,cdl);
time_ADMM=toc;

% Kmeans
tic
cfinal = ones(x*y,1);
[idx,~] = kmeans(ctemp,2);
cfinal(idx==1) = 0;
cfinal(idx==2) = 1;
temp = reshape(cfinal, x, y);
mean1 = mean(ctemp(cfinal==0));
mean2 = mean(ctemp(cfinal==1));
if mean2 < mean1

    map = zeros(x, y);
    map(temp==0) = 1;

else

    map = temp;

end
time_kmean=toc;

%% Evaluate

train_index = index;
out=map(:);
out(train_index)=[];

GT1D = reshape(gt,x*y,[]);
GT1D(train_index)=[];
[OA, kappa, pre, recall] = evaluate(out, GT1D);
time = time_DL+time_ADMM+time_kmean;

filename = sprintf('%s.mat',Data);
save(['./CODEresult/' filename],'map');

OA_all=[OA_all, OA];
kappa_all=[kappa_all, kappa];
precision_all=[precision_all, pre];
recall_all=[recall_all, recall];
time_all=[time_all, time];


end

OA_mean=mean(OA_all);
kappa_mean=mean(kappa_all);
pre_mean=mean(precision_all);
recall_mean=mean(recall_all);
time_mean=mean(time_all);

fprintf("====================CODE-HCD evaluation========================\n");
fprintf("\n");
fprintf("OA= %d\n",OA_mean);
fprintf("kappa= %d\n",kappa_mean);
fprintf("precision= %d\n",pre_mean);
fprintf("recall= %d\n",recall_mean);
fprintf("time= %d\n",time_mean);

%% Plot

normColor=@(R)max(min((R-mean(R(:)))/std(R(:)),2),-2)/3+0.5;

figure
subplot(1,3,1)
imshow(normColor(x1(:,:,[15 11 7])))
title('May 1, 2004')
subplot(1,3,2)
imshow(normColor(x2(:,:,[15 11 7])))
title('May 8, 2007')
subplot(1,3,3)
imshow(mat2gray(map))
title('Change map')



