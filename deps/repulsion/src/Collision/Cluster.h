#pragma once

namespace Collision
{
    
    template<typename I>
    struct Cluster // slim POD container to hold only the data relevant for the construction phase in the tree, before it is serialized
    {
    public:
        
        Cluster(){};

        Cluster( I thread_, I ID_, I begin_, I end_, I depth_ )
        :   thread(thread_)
        ,   ID(ID_)
        ,   begin(begin_)
        ,   end(end_)
        ,   depth(depth_)
        ,   max_depth(depth_)
        ,   descendant_count(0)
        ,   descendant_leaf_count(0)
        ,   left(nullptr)
        ,   right(nullptr)
        {}
        
        ~Cluster() { };
        
        I thread = -1;               // thread that created this cluster
        I ID = -1;                   // ID of cluster within this thread
        I begin = 0;                 // position of first primitive in cluster relative to array ordering
        I end = 0;                   // position behind last primitive in cluster relative to array ordering
        I depth = 0;                 // depth within the tree -- not absolutely necessary but nice to have for plotting images
        I max_depth = 0;             // maximal depth of all descendants of this cluster
        I descendant_count = 0;      // number of descendents of cluster, _this cluster included_
        I descendant_leaf_count = 0; // number of leaf descendents of cluster
        Cluster<I> *left = nullptr;        // left child
        Cluster<I> *right = nullptr;       // right child
        
    }; //Cluster
        
} // namespace Collision
