#pragma once

namespace Collision
{
    
    enum class BoundingVolumeType
    {
        AABB,
        AABB_LongestAxisSplit,
        AABB_MedianSplit,
        AABB_PreorderedSplit,
        OBB,
        OBB_MedianSplit,
        OBB_PreorderedSplit
    };
    
    enum class BlockSplitMethod
    {
        Parallel,
        Sequential,
        Recursive
    };
    
    enum class SplitMethod
    {
        Parallel,
        Sequential,
        Recursive
    };

} // namespace Collision
