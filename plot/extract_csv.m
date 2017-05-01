% loading the csv files generate by NetDissect
disp('extract semantics from raw data')
params = {};
params.networks_name = {'ResNet-152-model_places365','ResNet-152-model_imagenet','caffe_reference_places365','caffe_reference_places205','caffe_reference_imagenet'};
params.csv_files = {'ResNet-152-model_places365.csv','ResNet-152-model_imagenet.csv','caffe_reference_places365.csv','caffe_reference_places205.csv','caffe_reference_imagenet.csv'};

params.layers_name = network2layer(params.networks_name, 'networkprobe');


[unit_activations] = return_annotation( params );


thresh = 0.04; % IoU threshold to decide if an unit is a interpretable detector.
concepts = {'all','object','part','scene','material','texture', 'color',};
stat = {};
stat.networks_name = params.networks_name;
stat.layers_name = params.layers_name;
stat.ratio_detectors = zeros(numel(params.networks_name), numel(concepts));
stat.num_uniquedetectors = zeros(numel(params.networks_name), numel(concepts));
stat.num_detectors = zeros(numel(params.networks_name), numel(concepts));
stat.concepts = concepts;
stat.thresh_concepts = thresh;
fprintf('all\tnum\tunique\tobject\tnum\tunique\tpart\tnum\tunique\tscene\tnum\tunique\ttexture\tnum\tunique\tsurface\tnum\tunique\tcolor\tnum\tunique\tnetwork-layer\n')
for netID = 1:numel(params.networks_name)
    semantics_network = unit_activations{netID,3};

    scores_allconcept = str2double(semantics_network(:,2:2:end));
    headers_allconcept = unit_activations{netID,4}(2:2:end);
    [max_scores, max_idx] = max(scores_allconcept,[],2);
    num_unit = size(max_scores,1);
    
    num_detector_object = sum(scores_allconcept(:,1)>thresh);
    num_detector_object_unique = numel(unique(semantics_network(scores_allconcept(:,1)>thresh,1)));
    
    num_detector_objectpart = sum(scores_allconcept(:,6)>thresh);
    num_detector_objectpart_unique = numel(unique(semantics_network(scores_allconcept(:,6)>thresh,11)));
    
    num_detector_texture = sum(scores_allconcept(:,5)>thresh);
    num_detector_texture_unique = numel(unique(semantics_network(scores_allconcept(:,5)>thresh,9)));
    
    num_detector_surface = sum(scores_allconcept(:,3)>thresh);
    num_detector_surface_unique = numel(unique(semantics_network(scores_allconcept(:,3)>thresh,5)));
    
    num_detector_color = sum(scores_allconcept(:,2)>thresh);
    num_detector_color_unique = numel(unique(semantics_network(scores_allconcept(:,2)>thresh,3)));
  
    num_detector_scene = sum(scores_allconcept(:,4)>thresh);
    num_detector_scene_unique = numel(unique(semantics_network(scores_allconcept(:,4)>thresh,7)));

    num_detector = sum(max_scores>thresh);
    num_detector_unique = num_detector_object_unique+num_detector_objectpart_unique+num_detector_texture_unique+num_detector_surface_unique+num_detector_color_unique+num_detector_scene_unique;
    
    stat.num_detectors(netID,:) = [num_detector, num_detector_object, num_detector_objectpart, num_detector_scene, num_detector_surface, num_detector_texture, num_detector_color];
    
    stat.ratio_detectors(netID,:) = [num_detector/num_unit num_detector_object/num_unit num_detector_objectpart/num_unit num_detector_scene/num_unit num_detector_surface/num_unit num_detector_texture/num_unit num_detector_color/num_unit];
    stat.num_uniquedetectors(netID,:) = [num_detector_unique num_detector_object_unique num_detector_objectpart_unique num_detector_scene_unique num_detector_surface_unique num_detector_texture_unique num_detector_color_unique];
    fprintf('%.3f\t%d\t%d\t%.3f\t%d\t%d\t%.3f\t%d\t%d\t%.3f\t%d\t%d\t%.3f\t%d\t%d\t%.3f\t%d\t%d\t%.3f\t%d\t%d\t%d:%s-%s\n', num_detector/num_unit, num_detector, num_detector_unique, num_detector_object/num_unit, num_detector_object,num_detector_object_unique, num_detector_objectpart/num_unit, num_detector_objectpart, num_detector_objectpart_unique, num_detector_scene/num_unit, num_detector_scene,num_detector_scene_unique, num_detector_texture/num_unit, num_detector_texture,num_detector_texture_unique,num_detector_surface/num_unit,num_detector_surface,num_detector_surface_unique, num_detector_color/num_unit, num_detector_color,num_detector_color_unique, netID, params.networks_name{netID}, params.layers_name{netID});
end
save('semantics_samples.mat','stat');
