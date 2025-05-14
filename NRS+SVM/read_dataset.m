function [current_dataset_data,current_dataset_head,xlsx_file_name]=read_dataset(dirname)
%read the xlsx format dataset, so when you want to read data, please change the dataset into xlsx 
current_path=pwd;                                                                     %当前的文件夹
xlsx_file_name=uigetfile('.xlsx');                                                    %数据集
current_dataset_address=[current_path,dirname, xlsx_file_name];                   %数据集地址
[current_dataset_data,current_dataset_head]= xlsread(current_dataset_address);        %打开数据集
end

