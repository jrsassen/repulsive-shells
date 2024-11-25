#pragma once

namespace Collision {

    template<int AMB_DIM, typename Real, typename Int, typename SReal>
    void GJK_IntersectingQ_Batch
    (
        Int n,                                      // number of primitive pairs
        const PrimitiveSerialized<AMB_DIM,Real,Int,SReal> & P_, // prototype primitive
        SReal * const P_serialized_data,               // matrix of size n x P_.Size()
        const PrimitiveSerialized<AMB_DIM,Real,Int,SReal> & Q_, // prototype primitive
        SReal * const Q_serialized_data,               // matrix of size n x P_.Size()
        Int * const intersectingQ                    // vector of size n for storing the results
    )
    {
        tic("GJK_IntersectingQ_Batch");
        Int sub_calls = 0;

        Int thread_count;
        #pragma omp parallel
        {
            thread_count = omp_get_num_threads();
        }

#ifdef GJK_Report
        // Deactivating parallelization so that print() won't break.
        // cppcheck-suppress [redundantAssignment]
        thread_count = 1;
#endif

        valprint("Number of primitive pairs",n);
        valprint("Ambient dimension        ",AMB_DIM);
        valprint("thread_count             ",thread_count);
        print("First  primitive type     = "+P_.ClassName());
        print("Second primitive type     = "+Q_.ClassName());

        auto job_ptr = BalanceWorkLoad<Int>( n, thread_count );
        
        #pragma omp parallel for num_threads(thread_count) reduction( + : sub_calls )
        for( Int thread = 0; thread < thread_count; ++thread )
        {
            GJK_Algorithm<AMB_DIM,Real,Int> gjk;
            std::unique_ptr<PrimitiveSerialized<AMB_DIM,Real,Int,SReal>> P = P_.Clone();
            std::unique_ptr<PrimitiveSerialized<AMB_DIM,Real,Int,SReal>> Q = Q_.Clone();

            const Int i_begin = job_ptr[thread];
            const Int i_end   = job_ptr[thread+1];
            
            for( Int i = i_begin; i < i_end; ++i )
            {
                //print(\"======================================================\");\ valprint(\" i \",i);

                P->SetPointer( P_serialized_data, i );
                Q->SetPointer( Q_serialized_data, i );

                intersectingQ[i] = gjk.IntersectingQ( *P, *Q );
                
                sub_calls += gjk.SubCalls();
            }
        }

        print("GJK_IntersectingQ_Batch made " + ToString(sub_calls) + " subcalls for n = " + ToString(n) + " primitive pairs.");
        toc("GJK_IntersectingQ_Batch");
    }
    
    template<int AMB_DIM, typename Real, typename Int, typename SReal>
    void GJK_SquaredDistances_Batch
    (
        Int n,                                      // number of primitive pairs
        const PrimitiveSerialized<AMB_DIM,Real,Int,SReal> & P_, // prototype primitive
        SReal * const P_serialized_data,               // matrix of size n x P_.Size()
        const PrimitiveSerialized<AMB_DIM,Real,Int,SReal> & Q_, // prototype primitive
        SReal * const Q_serialized_data,               // matrix of size n x P_.Size()
        Real * const squared_dist                     // vector of size n for storing the squared distances
    )
    {
        tic("GJK_SquaredDistances_Batch");
        Int sub_calls = 0;

        Int thread_count;
        #pragma omp parallel
        {
            thread_count = omp_get_num_threads();
        }

#ifdef GJK_Report
        // Deactivating parallelization so that print() won't break.
        // cppcheck-suppress [redundantAssignment]
        thread_count = 1;
#endif

        valprint("Number of primitive pairs",n);
        valprint("Ambient dimension        ",AMB_DIM);
        valprint("thread_count             ",thread_count);
        print("First  primitive type     = "+P_.ClassName());
        print("Second primitive type     = "+Q_.ClassName());

        auto job_ptr = BalanceWorkLoad<Int>( n, thread_count );

        #pragma omp parallel for num_threads(thread_count) reduction( + : sub_calls )
        for( Int thread = 0; thread < thread_count; ++thread )
        {
            GJK_Algorithm<AMB_DIM,Real,Int> gjk;
            std::unique_ptr<PrimitiveSerialized<AMB_DIM,Real,Int,SReal>> P = P_.Clone();
            std::unique_ptr<PrimitiveSerialized<AMB_DIM,Real,Int,SReal>> Q = Q_.Clone();

            const Int i_begin = job_ptr[thread];
            const Int i_end   = job_ptr[thread+1];
            
            for( Int i = i_begin; i < i_end; ++i )
            {
                //print(\"======================================================\");\ valprint(\" i \",i);

                P->SetPointer( P_serialized_data, i );
                Q->SetPointer( Q_serialized_data, i );

                squared_dist[i] = gjk.SquaredDistance( *P, *Q );
                
                sub_calls += gjk.SubCalls();
            }
        }

        print("GJK_SquaredDistances_Batch made " + ToString(sub_calls) + " subcalls for n = " + ToString(n) + " primitive pairs.");
        toc("GJK_SquaredDistances_Batch");
    }
    
    template<int AMB_DIM, typename Real, typename Int, typename SReal>
    void GJK_Witnesses_Batch
    (
        Int n,                                       // number of primitive pairs
        const PrimitiveSerialized<AMB_DIM,Real,Int,SReal> & P_, // prototype primitive
        SReal * const P_serialized_data,               // matrix of size n x P_.Size()
         Real * const x,                               // matrix of size n x AMB_DIM for storing the witnesses in P
        const PrimitiveSerialized<AMB_DIM,Real,Int,SReal> & Q_, // prototype primitive
        SReal * const Q_serialized_data,               // matrix of size n x P_.Size()
         Real * const y                                // matrix of size n x AMB_DIM for storing the witnesses in P
    )
    {
        tic("GJK_Witnesses_Batch");
        
        Int sub_calls = 0;
        
        Int thread_count;
        
        #pragma omp parallel
        {
            thread_count = omp_get_num_threads();
        }

        auto job_ptr = BalanceWorkLoad<Int>( n, thread_count );
        
        valprint("Number of primitive pairs",n);
        valprint("Ambient dimension        ",AMB_DIM);
        valprint("thread_count             ",thread_count);
        print("First  primitive type     = "+P_.ClassName());
        print("Second primitive type     = "+Q_.ClassName());
        
        #pragma omp parallel for num_threads(thread_count) reduction( + : sub_calls )
        for( Int thread = 0; thread < thread_count; ++thread )
        {
            
            GJK_Algorithm<AMB_DIM,Real,Int> gjk;
            std::unique_ptr<PrimitiveSerialized<AMB_DIM,Real,Int,SReal>> P = P_.Clone();
            std::unique_ptr<PrimitiveSerialized<AMB_DIM,Real,Int,SReal>> Q = Q_.Clone();

            const Int i_begin = job_ptr[thread];
            const Int i_end   = job_ptr[thread+1];

            for( Int i = i_begin; i < i_end; ++i )
            {
                //print(\"======================================================\");\ valprint(\" i \",i);
                
                P->SetPointer( P_serialized_data, i );
                Q->SetPointer( Q_serialized_data, i );
                
//                valprint("P->Size()",P->Size());
//                valprint("Q->Size()",Q->Size());

                gjk.Witnesses( *P, x + AMB_DIM * i, *Q, y + AMB_DIM * i );
                
                sub_calls += gjk.SubCalls();
            }
        }

        print("GJK_Witnesses_Batch made " + ToString(sub_calls) + " subcalls for n = " + ToString(n) + " primitive pairs.");
        
        toc("GJK_Witnesses_Batch");
    }

} // namespe Collision
