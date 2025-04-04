clc
clear
close
surprise_list = 0 : 2 : 20;
corruption_num_list = surprise_list;
for z = 1 : length(corruption_num_list)
    for times = 1 : 1
        % load datasets
       name = ['./corrupted_datasets/surprise/surprise_38_corruption_',  num2str(corruption_num_list(z)),'_', num2str(times), '.mat'];
        load(name)
        [num_workers, num_tasks] = size(Y_obs);
        num_class = max(ground_truth);
        y = ground_truth;
        A = [];
        Z = zeros(num_tasks, num_class, num_workers);
        for i = 1 : num_class
            index = find(Y_obs_seperate(:,:,i) ~= 0);
            [worker_idx , task_idx] = ind2sub([num_workers, num_tasks], index);
            class_idx = i * ones(size(task_idx));
            A = [A; task_idx, worker_idx, class_idx];
            index_Z = sub2ind(size(Z), task_idx, class_idx, worker_idx);
            Z(index_Z) = 1;
        end
        
        B(:, 1) = [1 : num_tasks];
        B(:, 2) = ground_truth;
        n = num_tasks;
        m = num_workers;
        k = num_class;
        
        Nround = 1;
        mode = 1;
        
        error1_predict = zeros(1, Nround);
        error2_predict = zeros(1, Nround);

        valid_index = find(y > 0);

%======================SemMv===================================     
        opt.lambda1=10^3;
        opt.lambda2=10^-3;
        opt.lambda3=10^-3; 
        opt.num_class = num_class;
        [m,n] = size(Y_obs);
        Xdata=linspace(1,n,n);
        [ v,W,YY ] = SemMv( Xdata', Y_obs, opt );  
num_elements = numel(v);
num_select = round(0.2 * num_elements);
[~, sorted_indices] = sort(v);
min_indices = sorted_indices(1:num_select);
YY(min_indices, :) = 0;


           for i = 1:n
            for numc = 1:num_class
                posi = find(YY(:,i)==numc);
                sumc(numc) = sum(abs(YY(posi,i)));
            end
            cid = find(sumc==max(sumc));
            if length(cid)>1
                preY(i) = cid(2);
            else
                preY(i) = cid;
            end
        end

        acc  = Accuracy( preY, ground_truth );
        error_SW(times, z) = 1-acc;

             


        
    end
end



