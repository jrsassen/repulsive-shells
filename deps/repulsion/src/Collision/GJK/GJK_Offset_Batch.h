#pragma once

namespace Collision {

    template<int AMB_DIM, typename Real, typename Int, typename SReal>
    void GJK_Offset_IntersectingQ_Batch
    (
        Int n,                                       // number of primitive pairs
        const PrimitiveSerialized<AMB_DIM,Real,Int,SReal> & P_, // prototype primitive
              SReal * const P_serialized_data,         // matrix of size n x (1 + AMB_DIM + POINT_COUNT_1 x AMB_DIM)
        const Real * const P_off_set,                 // vector of size n storing the thickness of the primitive
        const PrimitiveSerialized<AMB_DIM,Real,Int,SReal> & Q_, // prototype primitive
              SReal * const Q_serialized_data,         // matrix of size n x (1 + AMB_DIM + POINT_COUNT_1 x AMB_DIM)
        const Real * const Q_off_set,                 // vector of size n storing the thickness of the primitive
              Int * const intersectingQ               // vector of size n for storing the result
    )
    {
        tic("GJK_Offset_IntersectingQ_Batch");
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

                intersectingQ[i] = gjk.Offset_IntersectingQ( *P, P_off_set[i], *Q, Q_off_set[i] );
                
                sub_calls += gjk.SubCalls();
            }
        }
        
        print("GJK_Offset_IntersectingQ_Batch made " + ToString(sub_calls) + " subcalls for n = " + ToString(n) + " primitive pairs.");
        toc("GJK_Offset_IntersectingQ_Batch");
    }
    
    template<int AMB_DIM,typename Real, typename Int, typename SReal>
    void GJK_Offset_SquaredDistances_Batch
    (
        Int n,                                             // number of primitive pairs
        const PrimitiveSerialized<AMB_DIM,Real,Int,SReal> & P_,       // prototype primitive
              SReal * const P_serialized_data,               // matrix of size n x (1 + AMB_DIM + POINT_COUNT_1 x AMB_DIM)
        const Real * const P_off_set,                       // vector of size n storing the thickness of the primitive
        const PrimitiveSerialized<AMB_DIM,Real,Int,SReal> & Q_,       // prototype primitive
              SReal * const Q_serialized_data,               // matrix of size n x (1 + AMB_DIM + POINT_COUNT_1 x AMB_DIM)
        const Real * const Q_off_set,                       // vector of size n storing the thickness of the primitive
              Real * const squared_dist                     // vector of size n for storing the squared distances
    )
    {
        tic("GJK_Offset_SquaredDistances_Batch");
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

                squared_dist[i] = gjk.Offset_SquaredDistance( *P, P_off_set[i], *Q, Q_off_set[i] );
                
                sub_calls += gjk.SubCalls();
            }
        }

        print("GJK_Offset_SquaredDistances_Batch made " + ToString(sub_calls) + " subcalls for n = " + ToString(n) + " primitive pairs.");
        toc("GJK_Offset_SquaredDistances_Batch");
    }
    
    template<int AMB_DIM,typename Real, typename Int, typename SReal>
    void GJK_Offset_Witnesses_Batch
    (
        Int n,                                             // number of primitive pairs
        const PrimitiveSerialized<AMB_DIM,Real,Int,SReal> & P_,       // prototype primitive
              SReal * const P_serialized_data,               // matrix of size n x (1 + AMB_DIM + POINT_COUNT_1 x AMB_DIM)
        const Real * const P_off_set,                       // vector of size n storing the thickness of the primitive
              Real * const x,                               // matrix of size n x AMB_DIM for storing the witnesses in P
        const PrimitiveSerialized<AMB_DIM,Real,Int,SReal> & Q_,       // prototype primitive
              SReal * const Q_serialized_data,               // matrix of size n x (1 + AMB_DIM + POINT_COUNT_1 x AMB_DIM)
        const Real * const Q_off_set,                       // vector of size n storing the thickness of the primitive
              Real * const y                                // matrix of size n x AMB_DIM for storing the witnesses in P
    )
    {
        tic("GJK_Offset_Witnesses_Batch");
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

                gjk.Offset_Witnesses( *P, P_off_set[i], x + AMB_DIM * i, *Q, Q_off_set[i], y + AMB_DIM * i );
                
                sub_calls += gjk.SubCalls();
            }

        }

        print("GJK_Offset_Witnesses_Batch made " + ToString(sub_calls) + " subcalls for n = " + ToString(n) + " primitive pairs.");
        toc("GJK_Offset_Witnesses_Batch");
    }

} // namespe Collision
