function [layers] = network2layer( networks, type_networks)
%NETWORK2FEATURE Summary of this function goes here
%   Detailed explanation goes here

layers = cell(numel(networks),1);
if strcmp(type_networks,'genericfeature')
    layerMap = get_layerMap_genericfeature();
elseif strcmp(type_networks, 'networkprobe')
    layerMap = get_layerMap_networkprobe();
end

try 
    for netID = 1:numel(networks)
        if strcmp(networks{netID}(1:11),'places_iter')
            layers{netID} = 'conv5';
        else
            layers{netID} = layerMap(networks{netID});
        end
    end
catch exception

    error(['no info for ' networks{netID}]);
end

end

function layerMap = get_layerMap_networkprobe()
    layerMap = containers.Map();
    layerMap('caffenet_random') = 'conv5';
    layerMap('caffe_reference_imagenet') = 'conv5';
    layerMap('caffe_reference_places205_bn') = 'conv5';
    layerMap('caffe_reference_places205') = 'conv5';
    layerMap('caffe_reference_places365') = 'conv5';
    layerMap('caffe_reference_places205_batchnorm') = 'conv5';
    layerMap('caffe_reference_imagenetplaces205') = 'conv5';
    layerMap('caffe_reference_places365_GAP') = 'conv5';
    layerMap('caffe_reference_places365_GAPplus') = 'conv5';
    layerMap('caffe_reference_places365_GAP1024') = 'conv5';
    layerMap('caffe_reference_places205_repeat1') = 'conv5';
    layerMap('caffe_reference_places205_repeat2') = 'conv5';
    layerMap('caffe_reference_places205_repeat3') = 'conv5';
    layerMap('caffe_reference_places205_repeat4') = 'conv5';
    layerMap('caffe_reference_places205_nodropout') = 'conv5';
    
    layerMap('vgg16_hybrid1365') = 'conv5_3';
    layerMap('vgg16_imagenet') = 'conv5_3';
    layerMap('vgg16_places205') = 'conv5_3';
    layerMap('vgg16_places365') = 'conv5_3';
    
    layerMap('weakly_audio') = 'conv5';
    layerMap('weakly_deepcontext') = 'conv5';
    layerMap('weakly_videoorder') = 'conv5';
    layerMap('weakly_videotracking') = 'conv5';
    layerMap('weakly_egomotion') = 'cls_conv5';
    layerMap('weakly_learningbymoving') = 'conv5';
    layerMap('weakly_objectcentric') = 'conv5';
    layerMap('weakly_solvingpuzzle') = 'conv5_s1';
    layerMap('weakly_colorization') = 'conv5'; 
    layerMap('weakly_splitbrain') = 'conv5'; 
    
    layerMap('googlenet_imagenet') = 'inception_5b-output';
    layerMap('googlenet_places205') = 'inception_5b-output';
    layerMap('googlenet_places365') = 'inception_5b-output';
    
    layerMap('ResNet-152-model_imagenet') = 'res5c';
    layerMap('ResNet-152-model_places365') = 'res5c';
    layerMap('ResNet-50-model_imagenet') = 'res5c';
    layerMap('resnet-152-torch-places365') = 'caffe.Eltwise_510';
    layerMap('resnet-152-torch-imagenet') = 'caffe.Eltwise_510';
    layerMap('rotation_020') = 'conv5';
    layerMap('rotation_040') = 'conv5';
    layerMap('rotation_060') = 'conv5';
    layerMap('rotation_080') = 'conv5';
    layerMap('rotation_100') = 'conv5';
end

function layerMap = get_layerMap_genericfeature()
    layerMap = containers.Map();
    layerMap('caffenet_random') = 'pool5';
    layerMap('caffe_reference_imagenet') = 'pool5';
    layerMap('caffe_reference_places205_bn') = 'pool5';
    layerMap('caffe_reference_places205') = 'pool5';
    layerMap('caffe_reference_places365') = 'pool5';
    layerMap('caffe_reference_places365_GAP') = 'pool5_gap';
    layerMap('caffe_reference_places205_batchnorm') = 'pool5';
    
    layerMap('caffe_reference_imagenetplaces205') = 'pool5';
    layerMap('vgg16_hybrid1365') = 'fc7';
    layerMap('vgg16_imagenet') = 'fc7';
    layerMap('vgg16_places205') = 'fc7';
    layerMap('vgg16_places365') = 'fc7';
    
    layerMap('weakly_audio') = 'pool5';
    layerMap('weakly_deepcontext') = 'pool5';
    layerMap('weakly_videoorder') = 'pool5';
    layerMap('weakly_videotracking') = 'pool5';
    layerMap('weakly_egomotion') = 'cls_pool5';
    layerMap('weakly_learningbymoving') = 'pool5';
    layerMap('weakly_objectcentric') = 'pool5';
    layerMap('weakly_solvingpuzzle') = 'pool5';
    layerMap('colorization_berkeley') = 'pool5'; 
    layerMap('googlenet_imagenet') = 'pool5/7x7_s1';
    layerMap('googlenet_places205') = 'pool5/7x7_s1';
    layerMap('googlenet_places365') = 'pool5/7x7_s1';
    layerMap('ResNet-152-model_imagenet') = 'pool5';
    layerMap('ResNet-152-model_places365') = 'pool5';
    layerMap('ResNet-50-model_imagenet') = 'pool5';
    layerMap('weakly_colorization') = 'pool5'; 
    layerMap('weakly_colorization_nors') = 'pool5'; 
    layerMap('weakly_splitbrain') = 'pool5'; 
    layerMap('weakly_splitbrain_nors') = 'pool5'; 
end
