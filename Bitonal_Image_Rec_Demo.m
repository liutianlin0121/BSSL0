%% Bitonal image reconstruction
% To run this program, you need CVX
% which can be downloaded from
% http://cvxr.com/cvx/

clear; clc;

%% Read image
X = imread('imgA2.png');
N = size(X,1);
X1 = X(:,:,1);

%% DFT and random sampling
W = dftmtx(N); %% DFT matrix
Xf = W*double(X1)*W;
x_orig = vec(X1);


Phi_orig = kron(W,W);
% random sampling
n = length(x_orig);
m = round(n/2); %% number of measurements
s = RandStream('mt19937ar','Seed',0);
idx = randperm(s,n,m);
Phi=Phi_orig(idx,:);

alphabet = [0,1];
p = [0.5,0.5];

%% Original Signal
mat_orig = mat(x_orig);

N = length(p);

%% Observation
xrand = x_orig(:)+0.1*randn(n,1); % additive noise

y = Phi*(xrand);

mat_rand = mat(xrand);
%% SAV

tic;

cvx_begin quiet
variable x_SAV(n,1);
minimize ( (1/2) * norm(x_SAV,1) + (1/2) * norm(x_SAV-1,1));    %(corrected!!! -- previously lambda and 1-lambda were switched) 
subject to
y == Phi*x_SAV
cvx_end

sol_SAV = (x_SAV >= 1/2);
mat_SAV = mat(sol_SAV);

time_SAV = toc;

%% Basis pursuit.

tic;

cvx_begin
variable x_bp(n)
minimize(norm(x_bp,1))
subject to
y==Phi*x_bp;
x_bp>=0;
cvx_end

sol_bp = (x_bp >= 1/2);
mat_bp = mat(sol_bp);

time_BP = toc;


%% Boxed Basis pursuit.

%tic;

%cvx_begin
%variable x_boxed_bp(n)
%minimize(norm(x_boxed_bp,1))
%subject to
%y==Phi*x_bp;
%0 <= x_boxed_bp <=1;
%cvx_end

%sol_boxed_bp = (x_boxed_bp >= 1/2);
%mat_boxed_bp = mat(sol_boxed_bp);

%time_boxed_BP = toc;

%%

tic;

cvx_begin
variable x_SN(n)
minimize(norm(x_SN,1) + 850 * norm(x_SN - 1/2,inf))
subject to
y==Phi*x_SN;
cvx_end

sol_SN = (x_SN >= 1/2);
mat_SN = mat(sol_SN);

time_SN = toc;

%% boxed reweighted SL0
tic;

x_BSSl0 = BSSl0(Phi, y, 0.5, 0.1, 0.9, 2, 3);
sol_BSSl0 = (x_BSSl0 >= 1/2);
mat_BSSl0 = mat(sol_BSSl0);
time_BSSl0 = toc;

%%

orig_resize = imresize(mat_orig,4,'nearest');
rand_resize = imresize(mat_rand,4,'nearest');
SAV_resize = imresize(mat_SAV,4,'nearest');
BP_resize = imresize(mat_bp,4,'nearest');
SN_resize = imresize(mat_SN,4,'nearest');
BSSl0_resize = imresize(mat_BSSl0,4,'nearest');


%%
figure; imshow(orig_resize,'Border','tight');
title('Original Image');

figure; imshow(rand_resize,'Border','tight');
title('Noised Image');

figure; imshow(SAV_resize,'Border','tight');
title('SAV');


figure; imshow(BP_resize,'Border','tight');
title('basis pursuit');


%figure; imshow(boxed_BP_resize,'Border','tight');
%title('basis pursuit');
figure; imshow(SN_resize,'Border','tight');
title('SN');

figure; imshow(BSSl0_resize,'Border','tight');
title('BSSl0');


%%
disp('Running time')
disp(['BP : ' num2str(time_BP) ',   ' 'SN : ' num2str(time_SN) ',   ' 'SAV : ' num2str(time_SAV) ',   ' 'Boxed Reweighted Smoothed l_0 : ' num2str(time_BSSl0)])


