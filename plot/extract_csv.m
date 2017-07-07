% loading the csv files generate by NetDissect
disp('extract semantics from raw data')
params = {};
params.networks_name = {'ResNet-152-model_places365','ResNet-152-model_imagenet','caffe_reference_places365','caffe_reference_places205','caffe_reference_imagenet'};
params.csv_files = {'ResNet-152-model_places365.csv','ResNet-152-model_imagenet.csv','caffe_reference_places365.csv','caffe_reference_places205.csv','caffe_reference_imagenet.csv'};

params.layers_name = network2layer(params.networks_name, 'networkprobe');


[unit_semantics] = return_annotation( params );


thresh = 0.04; % IoU threshold to decide if an unit is a interpretable detector.
concepts = {'object','part','scene','material','texture', 'color'};
indices_concepts = [1,6,4,3,5,2]; % (objet,part,scene,material,texture,color) as the column index inside the csv file
stat = {};
stat.networks_name = params.networks_name;
stat.layers_name = params.layers_name;
stat.ratio_detectors = zeros(numel(params.networks_name), numel(concepts)+1);
stat.num_uniquedetectors = zeros(numel(params.networks_name), numel(concepts)+1);
stat.num_detectors = zeros(numel(params.networks_name), numel(concepts)+1);
stat.concepts = ['all' concepts];
stat.thresh = thresh;
stat.indices_concepts = indices_concepts;


for netID = 1:numel(params.networks_name)
    semantics_network = unit_semantics{netID,3};

    scores_allconcept = str2double(semantics_network(:,2:2:end));
    [max_scores, max_idx] = max(scores_allconcept,[],2);
    num_unit = size(max_scores,1);
    
    num_detector_concepts = zeros(1,6);
    num_uniquedetector_concepts = zeros(1,6);
    for conceptID = 1:numel(indices_concepts)
        num_detector_overall = sum(scores_allconcept(:, indices_concepts(conceptID))>thresh);
        num_uniquedetector_overall = numel(unique(semantics_network(scores_allconcept(:,indices_concepts(conceptID))>thresh,indices_concepts(conceptID)*2-1)));
        num_detector_concepts(conceptID) = num_detector_overall;
        num_uniquedetector_concepts(conceptID) = num_uniquedetector_overall;
    end
    
    num_detector_overall = sum(max_scores>thresh);
    num_uniquedetector_overall = sum(num_detector_concepts);%num_detector_object_unique+num_detector_objectpart_unique+num_detector_texture_unique+num_detector_material_unique+num_detector_color_unique+num_detector_scene_unique;
    stat.num_detectors(netID,:) = [num_detector_overall num_detector_concepts];%[num_detector, num_detector_object, num_detector_objectpart, num_detector_scene, num_detector_material, num_detector_texture, num_detector_color];
    stat.ratio_detectors(netID,:) = stat.num_detectors(netID,:)./num_unit; %[num_detector/num_unit num_detector_object/num_unit num_detector_objectpart/num_unit num_detector_scene/num_unit num_detector_material/num_unit num_detector_texture/num_unit num_detector_color/num_unit];
    stat.num_uniquedetectors(netID,:) = [num_uniquedetector_overall num_uniquedetector_concepts];%[num_uniquedetector num_detector_object_unique num_detector_objectpart_unique num_detector_scene_unique num_detector_material_unique num_detector_texture_unique num_detector_color_unique];
end
save('semantics_samples.mat','stat','unit_semantics');
