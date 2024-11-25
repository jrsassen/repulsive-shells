#pragma once

namespace Repulsion
{
    
    enum class TreePercolationAlgorithm
    {
        Tasks,
        Sequential,
        Recursive
    };

    enum class InteractionKernels
    {
        Default,
        MinDist,
        Degenerate
    };
    
    enum class KernelType
    {
        FractionalOnly,
        HighOrder,
        LowOrder,
        SquaredDistance
    };
    
    static std::map<KernelType, std::string> KernelTypeName {
        {KernelType::FractionalOnly, "FractionalOnly"},
        {KernelType::HighOrder, "HighOrder"},
        {KernelType::LowOrder, "LowOrder"},
        {KernelType::SquaredDistance, "SquaredDistance"}
    };
    
} // namespace Repulsion
