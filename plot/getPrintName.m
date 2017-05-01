function list_output = getPrintName( list_featurename, type_print)
    if strcmp(type_print, 'feature')
        printMap = construct_dictionary_feature();
    elseif strcmp(type_print, 'semantics')
        printMap = construct_dictionary_semantics();
    elseif strcmp(type_print,'width')
        printMap = construct_dictionary_width();
    end
    list_output = cell(numel(list_featurename),1);
    
    for i = 1:numel(list_featurename)
        try
            list_output{i} = printMap(list_featurename{i});
        catch exception
            error(list_featurename{i})
        end
    end


end

function printMap = construct_dictionary_width()
    printMap = containers.Map();
    printMap('caffe_reference_places365_GAP_gap') = 'AlexNet-Places365-GAP';
    
end

function printMap = construct_dictionary_feature()

    printMap = containers.Map();
    printMap('ResNet-152-model_imagenet-pool5') = 'ResNet152-ImageNet';
    printMap('ResNet-50-model_imagenet-pool5') = 'ResNet50-ImageNet';
    printMap('googlenet_imagenet-pool5/7x7_s1') = 'GoogLeNet-ImageNet';
    printMap('ResNet-152-model_places365-pool5') = 'ResNet152-Places365';
    printMap('vgg16_hybrid1365-fc7') = 'VGG-Hybrid';
    printMap('vgg16_imagenet-fc7') = 'VGG-ImageNet';
    printMap('vgg16_places205-fc7') = 'VGG-Places205';
    printMap('vgg16_places365-fc7') = 'VGG-Places365';
    printMap('caffe_reference_imagenetplaces205-pool5') = 'AlexNet-Hybrid';
    printMap('googlenet_places205-pool5/7x7_s1') = 'GoogLeNet-Places205';
    printMap('googlenet_places365-pool5/7x7_s1') = 'GoogLeNet-Places365';
    printMap('caffe_reference_imagenet-pool5') = 'AlexNet-ImageNet';
    printMap('caffe_reference_places205_batchnorm-pool5') = 'AlexNet-Places205-BN';
    printMap('caffe_reference_places365_GAP-pool5_gap') = 'AlexNet-Places365-GAP';
    printMap('caffe_reference_places365-pool5') = 'AlexNet-Places365';
    printMap('caffe_reference_places205-pool5') = 'AlexNet-Places205';
    printMap('weakly_deepcontext-pool5') = 'context';
    printMap('weakly_colorization-pool5') = 'colorization';
    printMap('weakly_audio-pool5') = 'audio';
    printMap('weakly_splitbrain-pool5') = 'crosschannel';
    printMap('weakly_videotracking-pool5') = 'tracking';
    printMap('weakly_solvingpuzzle-pool5') = 'puzzle';
    printMap('weakly_objectcentric-pool5') = 'objectcentric';
    printMap('weakly_egomotion-cls_pool5') = 'egomotion';
    printMap('weakly_learningbymoving-pool5') = 'moving';
    printMap('caffenet_random-pool5') = 'AlexNet-random';
    printMap('weakly_videoorder-pool5') = 'frameorder';

end

function printMap = construct_dictionary_semantics()

    printMap = containers.Map();
    printMap('ResNet-152-model_imagenet') = 'ResNet152-ImageNet';
    printMap('ResNet-152-model_places365') = 'ResNet152-Places365';
    printMap('ResNet-50-model_imagenet') = 'ResNet50-ImageNet';
    printMap('resnet-152-torch-places365') = 'ResNet152-Places365';
    printMap('resnet-152-torch-imagenet') = 'ResNet152-ImageNet';
    printMap('vgg16_imagenet') = 'VGG-ImageNet';
    printMap('vgg16_places205') = 'VGG-Places205';
    printMap('vgg16_places365') = 'VGG-Places365';
    printMap('vgg16_hybrid1365') = 'VGG-Hybrid';
    printMap('googlenet_imagenet') = 'GoogLeNet-ImageNet';
    printMap('googlenet_places205') = 'GoogLeNet-Places205';
    printMap('googlenet_places365') = 'GoogLeNet-Places365';
    printMap('caffenet_random') = 'AlexNet-random';
    printMap('caffe_reference_imagenet') = 'AlexNet-ImageNet';
    printMap('caffe_reference_places205') = 'AlexNet-Places205';
    printMap('caffe_reference_imagenetplaces205') = 'AlexNet-Hybrid';
    printMap('caffe_reference_places205_batchnorm') = 'AlexNet-Places205-BatchNorm';
    printMap('caffe_reference_places365') = 'AlexNet-Places365';
    printMap('caffe_reference_places365_GAP') = 'AlexNet-Places365-GAP';
    printMap('caffe_reference_places365_GAP1024') = 'AlexNet-Places365-GAP1024';
    printMap('caffe_reference_places205_nodropout') = 'AlexNet-Places205-NoDropout';
    printMap('caffe_reference_places205_repeat1') = 'AlexNet-Places205-repeat1';
    printMap('caffe_reference_places205_repeat2') = 'AlexNet-Places205-repeat2';
    printMap('caffe_reference_places205_repeat3') = 'AlexNet-Places205-repeat3';
    printMap('caffe_reference_places205_repeat4') = 'AlexNet-Places205-repeat4';
    
    printMap('weakly_deepcontext') = 'context';
    printMap('weakly_audio') = 'audio';
    printMap('weakly_videoorder') = 'frameorder';
    printMap('weakly_videotracking') = 'tracking';
    printMap('weakly_egomotion') = 'egomotion';
    printMap('weakly_learningbymoving') = 'moving';
    printMap('weakly_objectcentric') = 'objectcentric';
    printMap('weakly_solvingpuzzle') = 'puzzle';
    printMap('weakly_colorization') = 'colorization';
    printMap('weakly_splitbrain') = 'crosschannel';

end