%% Reconstruct binary signals using:
%             (i)   Basis Pursuit(BP):
%                     minimize norm(z,1)
%                       subject to Phi*z = y.
%   
%             (ii)  Boxed BP
%                    minimize norm(z,1)
%                       subject to Phi*z = y, 0<= z <= 1 
%
%             (iii) Sum of Norms (SN):
%                    minimize norm(z,1) + lambda * norm(z- 1/2, infinity) 
%                       subject to Phi*z = y.
%         
%             (iv)  Sum of Absolute Values (SAV): 
%                     minimize p0 * norm(z,1) + (1-p0) * norm(z-1,1) 
%                       subject to Phi*z = y.
% 
%             (v) Boxed Smoothed l0:
%             (v) The proposed: Box-Constraint Sum of Smoothed l0 (BSSl0). 

% CVX is required: http://cvxr.com/cvx/
%
%
%
%  Edit History:
%  Initial Version: Sept. 2016
%  First edit: Tianlin Liu Dec. 2017
%  Second edit: Tianlin Liu Jan 2018
%  (c) Tianlin Liu
clear; clc; close all

%% Parameters

m=40; % number of measurements
n=100; % size of the original vector

%Nsim = 10000; % number of simulation. This will take a loooooong time! (approx. 5 days on my laptop) Please consider to uncomment the next line to get a quick impression instead!

%p_delta = 0.05; 

Nsim = 10; 
p_delta = 0.1; 



p_start = 0;
p_range = p_start:p_delta:1;
p_size = length(p_range);

boolean_success_BP = zeros(p_size,Nsim);
NSR_BP = zeros(p_size,Nsim);
time_BP = zeros(p_size,Nsim);

boolean_success_boxed_BP = zeros(p_size,Nsim);
NSR_boxed_BP = zeros(p_size,Nsim);
time_boxed_BP = zeros(p_size,Nsim);


boolean_success_SN = zeros(p_size,Nsim);
NSR_SN = zeros(p_size,Nsim);
time_SN = zeros(p_size,Nsim);


boolean_success_SAV = zeros(p_size,Nsim);
NSR_SAV = zeros(p_size,Nsim);
time_SAV = zeros(p_size,Nsim);


boolean_success_BSSl0 = zeros(p_size,Nsim);
NSR_BSSl0 = zeros(p_size,Nsim);
time_BSSl0 = zeros(p_size,Nsim);

boolean_success_Sl0 = zeros(p_size,Nsim);
NSR_Sl0 = zeros(p_size,Nsim);
time_Sl0 = zeros(p_size,Nsim);

boolean_success_boxed_Sl0 = zeros(p_size,Nsim);
NSR_boxed_Sl0 = zeros(p_size,Nsim);
time_boxed_Sl0 = zeros(p_size,Nsim);


boolean_success_OMP = zeros(p_size,Nsim);
NSR_OMP = zeros(p_size,Nsim);
time_OMP = zeros(p_size,Nsim);



%% Simulation
    j = 1;
    for p0=p_range
        messagetxt=sprintf('p = %f',p0);
        disp(messagetxt);
        
        rng('default')
        Phi=randn(m,n);
        
        for nsim = 1:Nsim
            messagetxt=sprintf('# of simulation = %d',nsim);
            disp(messagetxt);
            
            % Generate a test signal of cardinality S
            S = round((1 - p0).*n);
            x_orig=zeros(n,1);
            pos=randperm(n);
            x_orig(pos(1:S))= randsrc(S,1,[1]);
            
            % Measurement
            y=Phi*x_orig(:);
            
            % BP
            tic;
            cvx_begin quiet
            variable x_BP(n,1);
            minimize (norm(x_BP,1));     
            subject to
            y == Phi*x_BP
            cvx_end
    
            sol_BP = (x_BP >= 1/2);   % quantization of entries to {0,1}    
            time_BP(j,nsim) = toc;
            
            boolean_success_BP(j,nsim) = (nnz(sol_BP - x_orig) == 0);
            NSR_BP(j,nsim) = norm(sol_BP - x_orig)/norm(x_orig);

            % Boxed BP
            tic;
            cvx_begin quiet
            variable x_boxed_BP(n,1);
            minimize (norm(x_boxed_BP,1));     
            subject to
            y == Phi*x_boxed_BP
            0 <= x_boxed_BP <= 1
            cvx_end
    
            sol_boxed_BP = (x_boxed_BP >= 1/2);   % quantization of entries to {0,1}    
            time_boxed_BP(j,nsim) = toc;
            
            boolean_success_boxed_BP(j,nsim) = (nnz(sol_boxed_BP - x_orig) == 0);
            NSR_boxed_BP(j,nsim) = norm(sol_boxed_BP - x_orig)/norm(x_orig);

            
            % SN
            tic;
            cvx_begin quiet
            variable x_SN(n,1);
            minimize (norm(x_SN,1) + 100 * norm(x_SN - 1/2,inf));
            subject to
            y == Phi*x_SN
            cvx_end
    
            sol_SN = (x_SN >= 1/2);   % quantization of entries to {0,1}
            time_SN(j,nsim) = toc;
            
            boolean_success_SN(j,nsim) = (nnz(sol_SN - x_orig) == 0); 
            NSR_SN(j,nsim) = norm(sol_SN - x_orig)/norm(x_orig);            
            
            
            % SAV
            tic;
            cvx_begin quiet
            variable x_SAV(n,1);
            minimize (p0 * norm(x_SAV,1) + (1- p0) * norm(x_SAV-1,1));
            subject to
            y == Phi*x_SAV
            cvx_end
    
            sol_SAV = (x_SAV >= 1/2);   % quantization of entries to {0,1}
            time_SAV(j,nsim) = toc;
            
            boolean_success_SAV(j,nsim) = (nnz(sol_SAV - x_orig) == 0); 
            NSR_SAV(j,nsim) = norm(sol_SAV - x_orig)/norm(x_orig);
    
            % SL0
            tic;
            x_Sl0 = Sl0(Phi, y, 0.1, 0.5, 2, 1000);            
            sol_Sl0 =  (x_Sl0 >= 1/2);    % quantization of entries to {0,1}    
            time_Sl0(j,nsim) = toc;
            boolean_success_Sl0(j,nsim) = (nnz(sol_Sl0 - x_orig) == 0); 
            NSR_Sl0(j,nsim) = norm(sol_Sl0 - x_orig)/norm(x_orig);    
            
            % boxed SL0
            tic;
            x_boxed_Sl0 = boxed_SL0(Phi, y, (1-p0)*n, 0.1, 0.5, 2, 1000);            
            sol_boxed_Sl0 =  (x_boxed_Sl0 >= 1/2);    % quantization of entries to {0,1}    
            time_boxed_Sl0(j,nsim) = toc;
            boolean_success_boxed_Sl0(j,nsim) = (nnz(sol_boxed_Sl0 - x_orig) == 0); 
            NSR_boxed_Sl0(j,nsim) = norm(sol_boxed_Sl0 - x_orig)/norm(x_orig);    
            
            % OMP
            tic;
            x_OMP = omp(Phi, y , S);          
            sol_OMP =  (x_OMP >= 1/2);    % quantization of entries to {0,1}    
            time_OMP(j,nsim) = toc;
            boolean_success_OMP(j,nsim) = (nnz(sol_OMP - x_orig) == 0); 
            NSR_OMP(j,nsim) = norm(sol_OMP - x_orig)/norm(x_orig);          
            
                    
                          
            % Proposed: BSSl0
            tic;
            x_BSSl0 = BSSl0(Phi, y, p0, 0.1, 0.5, 2, 1000);            
            sol_BSSl0 =  (x_BSSl0 >= 1/2);    % quantization of entries to {0,1}    
            time_BSSl0(j,nsim) = toc;
            boolean_success_BSSl0(j,nsim) = (nnz(sol_BSSl0 - x_orig) == 0); 
            NSR_BSSl0(j,nsim) = norm(sol_BSSl0 - x_orig)/norm(x_orig);                  
           
        end
        
        j = j+1;
    end


%%

av_boolean_success_BP = sum(boolean_success_BP,2)/Nsim;
av_NSR_BP = sum(NSR_BP,2)/Nsim;
av_time_BP = sum(time_BP,2)/Nsim;

av_boolean_success_boxed_BP = sum(boolean_success_boxed_BP,2)/Nsim;
av_NSR_boxed_BP = sum(NSR_boxed_BP,2)/Nsim;
av_time_boxed_BP = sum(time_boxed_BP,2)/Nsim;

av_boolean_success_SN = sum(boolean_success_SN,2)/Nsim;
av_NSR_SN = sum(NSR_SN,2)/Nsim;
av_time_SN = sum(time_SN,2)/Nsim;

av_boolean_success_SAV = sum(boolean_success_SAV,2)/Nsim;
av_NSR_SAV = sum(NSR_SAV,2)/Nsim;
av_time_SAV = sum(time_SAV,2)/Nsim;



av_boolean_success_Sl0 = sum(boolean_success_Sl0,2)/Nsim;
av_NSR_Sl0 = sum(NSR_Sl0,2)/Nsim;
av_time_Sl0 = sum(time_Sl0,2)/Nsim;


av_boolean_success_boxed_Sl0 = sum(boolean_success_boxed_Sl0,2)/Nsim;
av_NSR_boxed_Sl0 = sum(NSR_boxed_Sl0,2)/Nsim;
av_time_boxed_Sl0 = sum(time_boxed_Sl0,2)/Nsim;



av_boolean_success_OMP = sum(boolean_success_OMP,2)/Nsim;
av_NSR_OMP = sum(NSR_OMP,2)/Nsim;
av_time_OMP = sum(time_OMP,2)/Nsim;


av_boolean_success_BSSl0 = sum(boolean_success_BSSl0,2)/Nsim;
av_NSR_BSSl0 = sum(NSR_BSSl0,2)/Nsim;
av_time_BSSl0 = sum(time_BSSl0,2)/Nsim;


messagetxt=sprintf('average BP time = %f seconds',mean(av_time_BP));
        disp(messagetxt);
messagetxt=sprintf('average Boxed BP time = %f seconds',mean(av_time_boxed_BP));
        disp(messagetxt);
messagetxt=sprintf('average SN time = %f seconds',mean(av_time_SN));
        disp(messagetxt);
messagetxt=sprintf('average SAV time = %f seconds',mean(av_time_SAV));
        disp(messagetxt);
messagetxt=sprintf('average SL0 time = %f seconds',mean(av_time_Sl0));
        disp(messagetxt);
messagetxt=sprintf('average boxed SL0 time = %f seconds',mean(av_time_boxed_Sl0));
        disp(messagetxt);

messagetxt=sprintf('average OMP time = %f seconds',mean(av_time_OMP));
        disp(messagetxt);

        
messagetxt=sprintf('average BSSL0 time = %f seconds',mean(av_time_BSSl0));
        disp(messagetxt);
        


av_boolean_failure_BP = 1 - av_boolean_success_BP;
av_boolean_failure_boxed_BP= 1 - av_boolean_success_boxed_BP;
av_boolean_failure_SN = 1 - av_boolean_success_SN;
av_boolean_failure_SAV = 1 - av_boolean_success_SAV;
av_boolean_failure_Sl0 = 1 - av_boolean_success_Sl0;
av_boolean_failure_boxed_Sl0 = 1 - av_boolean_success_boxed_Sl0;
av_boolean_failure_OMP = 1 - av_boolean_success_OMP;
av_boolean_failure_BSSl0 = 1 - av_boolean_success_BSSl0;

%%
figure(2); 


ax1 = subplot(4,1,1);

hold on 
line_av_boolean_failure_BP = plot( p_start:p_delta:1,av_boolean_failure_BP); 
line_av_boolean_failure_boxed_BP = plot( p_start:p_delta:1,av_boolean_failure_boxed_BP); 
line_av_boolean_failure_SN = plot( p_start:p_delta:1,av_boolean_failure_SN); 
line_av_boolean_failure_SAV = plot( p_start:p_delta:1,av_boolean_failure_SAV); 
line_av_boolean_failure_Sl0 = plot( p_start:p_delta:1,av_boolean_failure_Sl0); 
line_av_boolean_failure_boxed_Sl0 = plot( p_start:p_delta:1,av_boolean_failure_boxed_Sl0); 
line_av_boolean_failure_OMP = plot( p_start:p_delta:1,av_boolean_failure_OMP); 
line_av_boolean_failure_BSSl0 = plot( p_start:p_delta:1,av_boolean_failure_BSSl0); 

set(line_av_boolean_failure_BP, 'Color','black','Marker','+','linewidth',3);
set(line_av_boolean_failure_boxed_BP, 'Color','red','LineStyle',':','Marker','diamond','linewidth',3);
set(line_av_boolean_failure_SN,'Color','blue','LineStyle','--','linewidth',3);
set(line_av_boolean_failure_SAV, 'Color','magenta','LineStyle','-','Marker','*','linewidth',3);
set(line_av_boolean_failure_Sl0, 'Color','red','LineStyle','--','Marker','+','linewidth',3);
set(line_av_boolean_failure_boxed_Sl0, 'Color','cyan','Marker','+','linewidth',3);
set(line_av_boolean_failure_OMP, 'Color','yellow','linewidth',3);
set(line_av_boolean_failure_BSSl0, 'Color','green','linewidth',3);

       
%xlabel(ax1,'$p$','Interpreter','LaTex','FontSize',30);
ylabel(ax1,'FPR');
axis([p_start 1 0 1]);
set(gca,'FontSize',25);
hold off

ax2 = subplot(4,1,2);
hold on 

line_av_NSR_BP = plot( p_start:p_delta:1,(av_NSR_BP)); 
line_av_NSR_boxed_BP = plot( p_start:p_delta:1,(av_NSR_boxed_BP)); 
line_av_NSR_SN = plot( p_start:p_delta:1,(av_NSR_SN)); 
line_av_NSR_SAV = plot( p_start:p_delta:1,(av_NSR_SAV)); 
line_av_NSR_Sl0 = plot( p_start:p_delta:1,(av_NSR_Sl0)); 
line_av_NSR_boxed_Sl0 = plot( p_start:p_delta:1,(av_NSR_boxed_Sl0)); 
line_av_NSR_OMP = plot( p_start:p_delta:1,(av_NSR_OMP)); 

line_av_NSR_BSSl0 = plot( p_start:p_delta:1,(av_NSR_BSSl0)); 


set(line_av_NSR_BP, 'Color','black','Marker','+','linewidth',3);
set(line_av_NSR_boxed_BP, 'Color','red','LineStyle',':','Marker','diamond','linewidth',3);
set(line_av_NSR_SN,'Color','blue','LineStyle','--','linewidth',3);
set(line_av_NSR_SAV, 'Color','magenta','LineStyle','-','Marker','*','linewidth',3);
set(line_av_NSR_Sl0, 'Color','red','LineStyle','--','Marker','+','linewidth',3);
set(line_av_NSR_boxed_Sl0, 'Color','cyan','Marker','+','linewidth',3);
set(line_av_NSR_BSSl0, 'Color','green','linewidth',3);
set(line_av_NSR_OMP, 'Color','yellow','linewidth',3);


%xlabel(ax2,'$p$','Interpreter','LaTex','FontSize',30);
ylabel(ax2,'NSR'); 
axis([p_start 1 0 1]);

set(gca,'FontSize',25);
hold off

ax3 = subplot(4,1,3);

hold on 

line_av_time_BP = plot( p_start:p_delta:1,av_time_BP); 
line_av_time_boxed_BP = plot( p_start:p_delta:1,av_time_boxed_BP); 
line_av_time_SN = plot( p_start:p_delta:1,av_time_SN); 
line_av_time_SAV = plot( p_start:p_delta:1,av_time_SAV); 
line_av_time_Sl0 = plot( p_start:p_delta:1,av_time_Sl0); 
line_av_time_boxed_Sl0 = plot( p_start:p_delta:1,av_time_boxed_Sl0); 
line_av_time_OMP = plot( p_start:p_delta:1,av_time_OMP); 
line_av_time_BSSl0 = plot( p_start:p_delta:1,av_time_BSSl0); 


set(line_av_time_BP, 'Color','black','Marker','+','linewidth',3);
set(line_av_time_boxed_BP, 'Color','red','LineStyle',':','Marker','diamond','linewidth',3);
set(line_av_time_SN,'Color','blue','LineStyle','--','linewidth',3);
set(line_av_time_SAV, 'Color','magenta','LineStyle','-','Marker','*','linewidth',3);
set(line_av_time_Sl0, 'Color','red','LineStyle','--','Marker','+','linewidth',3);
set(line_av_time_boxed_Sl0, 'Color','cyan','Marker','+','linewidth',3);
set(line_av_time_OMP, 'Color','yellow','linewidth',3);
set(line_av_time_BSSl0, 'Color','green','linewidth',3);

xlabel(ax3,'$1-p$','Interpreter','LaTex','FontSize',30);
ylabel(ax3,'Run Time'); 
axis([p_start 1 0 1]);
set(gca,'FontSize',25);
hold off


hSub = subplot(4,1,4); 
hold on 

legend_BP = plot( 1, nan); 
legend_boxed_BP =  plot( 1, nan);
legend_SN =  plot( 1, nan);
legend_SAV = plot( 1, nan); 
legend_Sl0 =  plot( 1, nan); 
legend_boxed_Sl0 =  plot( 1, nan); 
legend_OMP =  plot( 1, nan); 
legend_BSSl0 =  plot( 1, nan); 


set(legend_BP, 'Color','black','Marker','+','linewidth',3);
set(legend_boxed_BP, 'Color','red','LineStyle',':','Marker','diamond','linewidth',3);
set(legend_SN,'Color','blue','LineStyle','--','linewidth',3);
set(legend_SAV, 'Color','magenta','LineStyle','-','Marker','*','linewidth',3);
set(legend_Sl0, 'Color','red','LineStyle','--','Marker','+','linewidth',3);
set(legend_boxed_Sl0, 'Color','cyan','Marker','+','linewidth',3);
set(legend_BSSl0, 'Color','green','linewidth',3);
set(legend_OMP, 'Color','yellow','linewidth',3);

lngd = legend([legend_BP legend_boxed_BP legend_SN legend_SAV legend_Sl0 legend_boxed_Sl0 legend_OMP legend_BSSl0],'BP', 'Boxed BP', 'SN', 'SAV', 'SL0','Boxed SL0', 'OMP', 'BSSL0',...
              'Location','southeast'); 

hold off
set(hSub, 'Visible', 'off');
set(lngd, 'Interpreter', 'latex', 'fontsize', 20);
tightfig;
