//==============================================================================
//
//  Copyright (c) 2017-2021 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include "SetBuilderOptions.hpp"

#include "SNPE/SNPE.hpp"
#include "DlContainer/IDlContainer.hpp"
#include "SNPE/SNPEBuilder.hpp"
#include "DlSystem/TensorShape.hpp"
#include "DlSystem/TensorShapeMap.hpp"
#include <iostream>

std::unique_ptr<zdl::SNPE::SNPE> setBuilderOptions(std::unique_ptr<zdl::DlContainer::IDlContainer> & container,
                                                   zdl::DlSystem::Runtime_t runtime,
                                                   zdl::DlSystem::RuntimeList runtimeList,
                                                   zdl::DlSystem::UDLBundle udlBundle,
                                                   bool useUserSuppliedBuffers,
                                                   zdl::DlSystem::PlatformConfig platformConfig,
                                                   bool useCaching)
{
    std::unique_ptr<zdl::SNPE::SNPE> snpe;
    zdl::SNPE::SNPEBuilder snpeBuilder(container.get());

    if(runtimeList.empty())
    {
        runtimeList.add(runtime);
    }

    snpe = snpeBuilder.setOutputLayers({})
       .setRuntimeProcessorOrder(runtimeList)
       .setPerformanceProfile(zdl::DlSystem::PerformanceProfile_t::HIGH_PERFORMANCE)
       //.setUdlBundle(udlBundle)
       .setUseUserSuppliedBuffers(useUserSuppliedBuffers)
       .setPlatformConfig(platformConfig)
       .setInitCacheMode(useCaching)
       .build();

    zdl::DlSystem::StringList names = snpe->getInputTensorNames();
    for (auto ptr=names.begin(); ptr < names.end(); ptr ++) {
        fprintf(stdout, "[Input layer]%s\n", *ptr);
        //const auto &tensorShape = *(snpe->getInputDimensions(*ptr));
        //fprintf(stdout, "[Input dim]%d\n", static_cast<int>(tensorShape.getDimensions()[0]));
    }
    
    return snpe;
}

std::unique_ptr<zdl::SNPE::SNPE> setBuilderOptions2(std::unique_ptr<zdl::DlContainer::IDlContainer> & container,
                                                   zdl::DlSystem::Runtime_t runtime,
                                                   zdl::DlSystem::RuntimeList runtimeList,
                                                   zdl::DlSystem::UDLBundle udlBundle,
                                                   bool useUserSuppliedBuffers,
                                                   zdl::DlSystem::PlatformConfig platformConfig,
                                                   bool useCaching, const int width, const int height)
{
    std::unique_ptr<zdl::SNPE::SNPE> snpe;
    zdl::SNPE::SNPEBuilder snpeBuilder(container.get());

    if(runtimeList.empty())
    {
        runtimeList.add(runtime);
    }

    zdl::DlSystem::TensorShapeMap inputShapeMap;
    zdl::DlSystem::Dimension inputDims[4] = {1, static_cast<zdl::DlSystem::Dimension>(height), static_cast<zdl::DlSystem::Dimension>(width), 3};
    zdl::DlSystem::TensorShape inputShape(inputDims, 4);
    inputShapeMap.add("input_1:0", inputShape);

    fprintf(stdout, "setBuilderOptions width=%d height=%d\n", width, height);

    snpe = snpeBuilder.setOutputLayers({})
       .setRuntimeProcessorOrder(runtimeList)
       .setPerformanceProfile(zdl::DlSystem::PerformanceProfile_t::HIGH_PERFORMANCE)
       //.setUdlBundle(udlBundle)
       .setUseUserSuppliedBuffers(useUserSuppliedBuffers)
       .setPlatformConfig(platformConfig)
       .setInitCacheMode(useCaching)
       .setInputDimensions(inputShapeMap)
       .build();
    return snpe;
}