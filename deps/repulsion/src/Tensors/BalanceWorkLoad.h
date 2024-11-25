#pragma once

namespace Tensors
{
    // TODO: Make the job pointers multiples of CACHE_LINE_WIDTH
    
    template<typename I>
    Tensor1<I,I> BalanceWorkLoad( const I job_count, const I thread_count )
    {
        auto job_ptr = Tensor1<I,I>( thread_count + 1);
        for( I k = 0; k < thread_count+1; ++k )
        {
            job_ptr[k] = (k * job_count) / thread_count;
        }
        return job_ptr;
    }
    
    template<typename I>
    Tensor1<I,I> BalanceWorkLoad(
        const I job_count,
        I const * const costs,
        const I thread_count,
        bool accumulate = true
    )
    {
        // This function reads in a list job_acc_costs of accumulated costs, then allocates job_ptr as a vector of size thread_count + 1, and writes the work distribution to it.
        // Aasigns threads to consecutive chunks jobs, ..., job_ptr[k+1]-1 of jobs.
        // Uses a binary search to find the chunk boundaries.
        // The cost of the i-th job is job_acc_costs[i+1] - job_acc_costs[i].
        // The cost of the k-th thread goes from job no job_ptr[k] to job no job_ptr[k+1] (as always in C/C++, job_ptr[k+1] points _after_ the last job.
    
        
        ptic("BalanceWorkLoad");
        
        //        valprint("job_count",job_count);
        
//        DUMP(job_count);
//        auto costs_ = Tensor1<I,I> (costs,job_count+1);
//        DUMP(costs_);
        
        
        I * acc = nullptr;
        if( accumulate )
        {
            safe_alloc( acc, job_count + 1 );
            acc[0] = 0;
            I sum = 0;
            for( I i = 0 ; i < job_count; ++i )
            {
                sum += costs[i];
                acc[ i + 1 ] = sum;
            }
        }
        else
        {
            acc = const_cast<I *>(costs);
        }

//        auto acc_ = Tensor1<I,I> (acc,job_count+1);
//        print("acc_ = " + acc_.ToString());
        
        Tensor1<I,I> job_ptr ( thread_count + 1, 0 );
        job_ptr[0] = 0;
        job_ptr[thread_count] = job_count;

        const I naive_chunk_size = (job_count + thread_count - 1) / thread_count;
        
        const I total_cost = acc[job_count];
        
        const I per_thread_cost = (total_cost + thread_count - 1) / thread_count;
        

        // binary search for best work load distribution
        
        //  TODO: There is quite a lot false sharing in this loop... so better not parallelize it.
        for( I thread = 0; thread < thread_count - 1; ++thread)
        {
//            std::cout << "\n #### thread = " << thread << std::endl;
            // each thread (other than the last one) is require to have at least this accumulated cost
            I target = std::min( total_cost, per_thread_cost * (thread + 1) );
            I pos;
            // find an index a such that b_row_acc_costs[ a ] < target;
            // taking naive_chunk_size * thread as initial guess, because that might be nearly correct for costs that are evenly distributed over the block rows
            pos = thread + 1;
            I a = std::min(job_count, naive_chunk_size * pos);
            while( acc[a] >= target )
            {
                --pos;
                a = std::min(job_count, naive_chunk_size * pos);
            };
            
            // find an index  b such that b_row_acc_costs[ b ] >= target;
            // taking naive_chunk_size * (thread + 1) as initial guess, because that might be nearly correct for costs that are evenly distributed over the block rows
            pos = thread + 1;
            I b = std::min(job_count, naive_chunk_size * pos);
            while( (b < job_count) && (acc[b] < target) )
            {
                ++pos;
                b = std::min(job_count, naive_chunk_size * pos);
            };

            // binary search until
            I c;
            while( b > a + 1 )
            {
                c = a + (b-a)/2;
                if( acc[c] > target )
                {
                    b = c;
                }
                else
                {
                    a = c;
                }
            }
            job_ptr[thread + 1] = b;
        }
        
        if( accumulate )
        {
            safe_free(acc);
        }
        
        if( total_cost <=0 )
        {
//            wprint("BalanceWorkLoad: Total cost is 0.");
            job_ptr = Tensor1<I,I>(2,0);
        }
        
        ptoc("BalanceWorkLoad");
        
        return job_ptr;
    } // BalanceWorkLoad
    
    template<typename I>
    Tensor1<I,I> BalanceWorkLoad(
        const Tensor1<I,I> & costs,
        const I thread_count,
        const bool accumulate = true
    )
    {
        if( costs.Size() > 0 )
        {
            if( accumulate )
            {
                return BalanceWorkLoad<I>( costs.Dimension(0),   costs.data(), thread_count, true  );
            }
            else
            {
                return BalanceWorkLoad<I>( costs.Dimension(0)-1, costs.data(), thread_count, false );
            }
        }
        else
        {
            wprint("BalanceWorkLoad: 'costs' is an empty list. ");
            return Tensor1<I,I>(static_cast<I>(1),static_cast<I>(0));
        }
    }
    
} // namespace Tensors
