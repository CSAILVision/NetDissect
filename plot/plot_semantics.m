% sample script to plot the summary of detector numbers for all the
% networks
% run extract_csv.m first to extract the semantics from the raw csv if you
% jush finish running network dissection

clear
load('semantics_cvpr_release.mat');

selectIDX_networks = [1:numel(stat.networks_name)];

stat.networks_name = stat.networks_name(selectIDX_networks);
stat.layers_name = stat.layers_name(selectIDX_networks);
stat.ratio_detectors = stat.ratio_detectors(selectIDX_networks,:);
stat.num_uniquedetectors = stat.num_uniquedetectors(selectIDX_networks,:);
stat.num_detectors = stat.num_detectors(selectIDX_networks,:);

sum_uniquedetectors = sum(stat.num_uniquedetectors(:,2:end),2);
[value_sort, idx_sort] = sort(sum_uniquedetectors,'descend');
stat.networks_name = stat.networks_name(idx_sort);
stat.layers_name = stat.layers_name(idx_sort);
stat.ratio_detectors = stat.ratio_detectors(idx_sort,:);
stat.num_uniquedetectors = stat.num_uniquedetectors(idx_sort,:);
stat.num_detectors = stat.num_detectors(idx_sort,:);

disp(stat);
networks_print = getPrintName(stat.networks_name,'semantics');

% figure,
% %plot([1:size(stat.ratio_detectors,1)], stat.ratio_detectors', '--o'),
% bar(stat.ratio_detectors, 'stacked'),
% legend(stat.concepts),title('Ratio of detectors');
% xticks([1:size(stat.ratio_detectors,1)])
% xticklabels(networks_print),xtickangle(45)

figure, 
subplot(1,2,1);
bar( stat.num_uniquedetectors(:,2:end),'stacked'),
legend(stat.concepts(2:end)),title('Number of unique detectors');
xticks([1:size(stat.ratio_detectors,1)])
xticklabels(networks_print), xtickangle(45)

subplot(1,2,2);
bar( stat.num_detectors(:,2:end),'stacked'),
legend(stat.concepts(2:end)),title('Number of detectors');
xticks([1:size(stat.num_detectors,1)])
xticklabels(networks_print), xtickangle(45)