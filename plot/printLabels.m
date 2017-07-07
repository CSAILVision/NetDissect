function [list_output] = printLabels( list_input)
% remove some legacy signs in the names of concepts
list_output = cell(numel(list_input),1);
for i = 1:numel(list_input)
    tmp = list_input{i};
    tmp(1:end-2) = strrep(tmp(1:end-2),'_',' ');
    tmp(1:end-2) = strrep(tmp(1:end-2),'-',' ');
    tmp = strrep(tmp,'-s','');
    tmp = strrep(tmp,'-c','');
    list_output{i} = tmp;
end
    

end

