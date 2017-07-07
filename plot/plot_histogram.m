% plot the histogram of all the detectors the network dissection identifies
% for the given networks. 
clear
load('semantics_samples.mat');
thresh = stat.thresh;
concepts = stat.concepts(2:end);% the first one is the all the detector
indices_concepts = stat.indices_concepts;

concepts_select = {'object','texture','scene'}; % select the concepts to plot
[concepts_select, ia, ib] = intersect(concepts_select, concepts);
indices_concepts_select = indices_concepts(ib);

for netID = 1:numel(stat.networks_name)
    figure
    semantics_network = unit_semantics{netID,3};
    num_unit = size(semantics_network,1);
    scores_allconcept = str2double(semantics_network(:,2:2:end));

    for conceptID = 1:numel(concepts_select)
        detector_concept = semantics_network(scores_allconcept(:,indices_concepts_select(conceptID))>thresh, indices_concepts_select(conceptID)*2-1);

        [unique_data,junk,ind] = unique(detector_concept);
        freq_unique_data = histc(ind,1:numel(unique_data));
        [value_sort, idx_sort] = sort(freq_unique_data,'descend');
        uniquedetectors = unique_data(idx_sort);
        freq_detectors = value_sort;

        subplot(numel(concepts_select),1,conceptID);
        set(gcf,'Color',[1 1 1]);
        
        bar( freq_detectors,'stacked'),
        title(sprintf('Histogram of %s detectors', concepts_select{conceptID}));
        xticks([1:numel(uniquedetectors)])
        xticklabels(printLabels(uniquedetectors)), xtickangle(45)

        set(gca,'FontSize',20);
        xlim(gca,[0 numel(uniquedetectors)+1])  
    end
    xlabel(strrep(stat.networks_name{netID},'_','-'));
end