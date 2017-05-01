function [unit_activations] = return_annotation(params)
% extract semantics from the raw csv file.

unit_activations = cell(numel(params.networks_name),4);

for i = 1:numel(params.csv_files)

    semantics_file = fullfile(params.csv_files{i});
    fprintf('processing %s\n', semantics_file)
    
    data = fopen(semantics_file);
    csv_data = textscan(data,'%s','Delimiter','\n');
    semantics_headers = textscan(csv_data{1}{1},'%s','Delimiter',',');
    semantics_headers = semantics_headers{1};
    num_units = numel(csv_data{1})-1;
    semantics_values = cell(num_units, numel(semantics_headers));
    for unitID = 1:num_units
        C = textscan(csv_data{1}{unitID+1},'%s','Delimiter',',');
        D = C{1};
        semantics_values(str2double(D{1}),:) = D;
    end
    unit_activations{i, 1} = params.networks_name{i}; 
    unit_activations{i, 2} = params.layers_name{i}; 
    unit_activations{i, 3} = select_semantics(semantics_values);
    unit_activations{i, 4} = semantics_headers; 
    fclose(data);
end

end

function semantics_new = select_semantics(semantics)
    tmp_1 = [5:5:size(semantics,2)];
    tmp_2 = [9:5:size(semantics,2)];
    index_concepts = [tmp_1, tmp_2];
    index_concepts = sort(index_concepts,'ascend');
    semantics_new = semantics(:, index_concepts);
end
