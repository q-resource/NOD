folder = '/public/home/zhairq/imagen_followup2';
%folder = '/public/home/zhairq/imagen-followup2/ses-followup2';
subfolders = dir(fullfile(folder, 'sub-0*'));
%Mask=y_ReadAll('/public/home/zhairq/imagen_eft_code_data/AAL_61x73x61_YCG.nii');
%Frontal_Inf_Oper_R=y_ReadAll('/public/home/zhairq/imagen_eft_code_data/Frontal_Inf_Oper_R_facial_recg_30_cutoff.nii');
Fusiform_L=y_ReadAll('/public/home/zhairq/imagen_eft_code_data/Fusiform_L_facial_recg_35_cutoff.nii');

%Fusiform_R=y_ReadAll('/public/home/zhairq/imagen_eft_code_data/fusiform_R_cutoff_facial_recg_34.5_mask.nii');
load('filtered_subfolders.mat','filtered_subfolders');
best_base_x = nan(length(subfolders), 1);
best_base_xc = nan(length(subfolders), 1);
space_vector_new_zero_point_base = csvread('space_vector_new_zero_point_combine_model_Fusiform_L_35_cutoff.csv');

space_vector_new_zero_point_std = csvread('space_vector_new_zero_point_std_combine_model_Fusiform_L_35_cutoff.csv');

% 
% parpool('local', 4); % specify the number of workers to use
% 
% % attach necessary files to the workers
% addAttachedFiles(gcp, {'y_ReadAll.m', 'filtered_subfolders.mat', ...
%     'space_vector_new_zero_point_first_model_individual_Temporal_Pole_Sup_L.csv', ...
%     'space_vector_new_zero_point_std_first_model_individual_Temporal_Pole_Sup_L.csv'});
%results = struct();
parfor i = 1:length(subfolders)
    if subfolders(i).isdir && ~strcmp(subfolders(i).name,'.') && ~strcmp(subfolders(i).name,'..')
        if any(strcmp(subfolders(i).name, filtered_subfolders)) 
          subfolder_path = fullfile(subfolders(i).folder, subfolders(i).name);
          path1='con_angry.nii';
          path2='con_neutral.nii';
          path3='con_happy.nii';
          [sub_dataa,~,~,headera]=y_ReadAll(fullfile(subfolder_path, path1));
          [sub_datan,~,~,headern]=y_ReadAll(fullfile(subfolder_path, path2));
          [sub_datah,~,~,headerh]=y_ReadAll(fullfile(subfolder_path, path3));
          idx1=~isnan(sub_dataa);
          idx2=~isnan(sub_datan);
          idx3=~isnan(sub_datah);
          %y1=sub_dataa((Mask(:)==11)&(idx1(:))&(idx2(:))&(idx3(:)));
          y1=sub_dataa((Fusiform_L(:)==1)&(idx1(:))&(idx2(:))&(idx3(:)));
          y2=sub_datan((Fusiform_L(:)==1)&(idx1(:))&(idx2(:))&(idx3(:)));
          y3=sub_datah((Fusiform_L(:)==1)&(idx1(:))&(idx2(:))&(idx3(:)));
          y1v=y1(:);
          y2v=y2(:);
          y3v=y3(:);
          num=sum(sum(sum(y1~=0)));
          x = 0:0.02:5;
          xc = 0:0.02:5;
        cor_search_value = zeros(length(x), length(xc));
        cor_search_value_abs = zeros(length(x), length(xc));
        beta1 = zeros(num, 1);
        beta2 = zeros(num, 1);
        for k =1:length(x)
            for m =1:length(xc)
            for j =1:num
                yreg=[y1v(j,:);y2v(j,:);y3v(j,:)];
                X=[-x(k),1,1;0,0,1;1,xc(m),1];
                b=inv(X'*X)*X'*yreg;
                beta1(j)=b(1);
                beta2(j)=b(2);
            end
%             cor_search_value(k,:)=corr(beta1,beta2);
             cor_search_value(k,m)=corr(beta1,beta2);
             cor_search_value_abs(k,m)=abs(corr(beta1,beta2)-space_vector_new_zero_point_base(k,m));
            end
        end
        %cor_search_value=cor_search_value(:);
%         min_idx = find((cor_search_value > (-0.07645)) & (cor_search_value < 0.182), 1, 'first');%还可以根据模拟找到精确解，每个x都有一个新零点
%         if length(min_idx)==0
%             best_base(i)=0;
%         else
%             best_base(i)=-x(min_idx);
%         end
         [min_value,min_idx]=min(cor_search_value_abs(:));
         [idx_row, idx_col] = find(cor_search_value_abs == min_value);
%          best_base(i)=x(min_idx)
          if (cor_search_value(idx_row(1),idx_col(1))>(space_vector_new_zero_point_base(idx_row(1),idx_col(1))-2*space_vector_new_zero_point_std(idx_row(1),idx_col(1))))&&(cor_search_value(idx_row(1),idx_col(1))<(space_vector_new_zero_point_base(idx_row(1),idx_col(1))+2*space_vector_new_zero_point_std(idx_row(1),idx_col(1))))    
%           if (cor_search_value(min_idx)>(space_vector_new_zero_point_base(min_idx)-0.2))&&(cor_search_value(min_idx)<(space_vector_new_zero_point_base(min_idx)+0.2))
             best_base_x(i)=-x(idx_row(1));
             best_base_xc(i)=x(idx_col(1));
%              results(i).name = subfolders(i).name;
%              results(i).best_base_x = best_base_x(i);
%              results(i).best_base_xc = best_base_xc(i);
              output_file1 = fullfile(subfolder_path, 'Fusiform_L_individual_combine_model_best_base_x.txt');
              dlmwrite(output_file1, best_base_x(i));
                output_file2 = fullfile(subfolder_path, 'Fusiform_L_individual_combine_model_best_base_xc.txt');
               dlmwrite(output_file2, best_base_xc(i));
          else
             best_base_x(i)=-100;
             best_base_xc(i)=-100;
          end
        end
       end

end
best_base_x=best_base_x(~isnan(best_base_x));
best_base_xc=best_base_xc(~isnan(best_base_xc));
save('Fusiform_L_35_cutoff_individual_combine_model_best_base_x.mat','best_base_x');
save('Fusiform_L_35_cutoff_individual_combine_model_best_base_xc.mat','best_base_xc');
best_base_x;
best_base_xc;
%save('Temporal_Mid_L_individual_combine_model_two_best_base.mat', 'results');



