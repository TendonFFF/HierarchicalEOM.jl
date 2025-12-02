    @testitem "BlockSparseRowMatrix basics" tags=[:BSR] begin
        using SparseArrays
        # Create a simple BSR matrix
        block_size = 2
        nrows = 2
        ncols = 2
        
        # Create two unique blocks
        block1 = sparse([1.0+0im 2.0; 3.0 4.0])
        block2 = sparse([5.0+0im 6.0; 7.0 8.0])
        
        unique_blocks = [block1, block2]
        block_rowval = [1, 2, 1]  # blocks at (1,1), (2,1), (1,2)
        block_colptr = [1, 3, 4]  # column 1 has 2 blocks, column 2 has 1 block
        block_indices = [1, 2, 1]  # use block1, block2, block1 (deduplication!)
        
        bsr = BlockSparseRowMatrix(block_size, nrows, ncols, unique_blocks, block_rowval, block_colptr, block_indices)
        
        @test size(bsr) == (4, 4)
        @test nnz_blocks(bsr) == 3
        @test n_unique_blocks(bsr) == 2
        @test memory_savings(bsr) == 2/3  # 2 unique blocks out of 3 total
        
        # Test element access
        @test bsr[1, 1] == 1.0 + 0im
        @test bsr[1, 2] == 2.0 + 0im
        @test bsr[2, 1] == 3.0 + 0im
        
        # Test getblock
        @test getblock(bsr, 1, 1) == block1
        @test getblock(bsr, 2, 1) == block2
        @test getblock(bsr, 1, 2) == block1  # Same as (1,1) due to deduplication
        @test getblock(bsr, 2, 2) === nothing  # Zero block
        
        # Test conversion to sparse
        sparse_mat = to_sparse(bsr)
        @test size(sparse_mat) == (4, 4)
        @test sparse_mat[1, 1] == 1.0 + 0im
    end
    
    @testitem "BSRBuilder" tags=[:BSR] begin
        using SparseArrays
        block_size = 2
        nrows = 3
        ncols = 3
        
        builder = BSRBuilder(block_size, nrows, ncols)
        
        # Add some blocks
        block1 = sparse([1.0+0im 0.0; 0.0 1.0])
        block2 = sparse([2.0+0im 0.0; 0.0 2.0])
        
        add_block!(builder, 1, 1, block1)
        add_block!(builder, 2, 2, block1)  # Same as (1,1) - will be deduplicated
        add_block!(builder, 3, 3, block1)  # Same as (1,1) - will be deduplicated
        add_block!(builder, 1, 2, block2)
        
        bsr = build_bsr_matrix(builder, verbose=false)
        
        @test size(bsr) == (6, 6)
        @test nnz_blocks(bsr) == 4
        @test n_unique_blocks(bsr) <= 2  # Should have deduplicated the identical blocks
        
        # Verify the blocks are correctly placed
        @test getblock(bsr, 1, 1) !== nothing
        @test getblock(bsr, 2, 2) !== nothing
        @test getblock(bsr, 3, 3) !== nothing
        @test getblock(bsr, 1, 2) !== nothing
    end
    
    @testitem "BSROperator" tags=[:BSR] begin
        using SparseArrays
        using LinearAlgebra
        import SciMLOperators
        # Create a simple BSR matrix
        block_size = 2
        nrows = 2
        ncols = 2
        
        block1 = sparse([1.0+0im 0.0; 0.0 1.0])
        block2 = sparse([2.0+0im 0.0; 0.0 2.0])
        
        unique_blocks = [block1, block2]
        block_rowval = [1, 2]
        block_colptr = [1, 2, 3]
        block_indices = [1, 2]
        
        bsr = BlockSparseRowMatrix(block_size, nrows, ncols, unique_blocks, block_rowval, block_colptr, block_indices)
        op = BSROperator(bsr)
        
        @test size(op) == (4, 4)
        @test eltype(op) == ComplexF64
        @test SciMLOperators.isconstant(op) == true
        
        # Test matrix-vector multiplication
        x = ones(ComplexF64, 4)
        y = op * x
        
        @test length(y) == 4
        @test y[1] ≈ 1.0
        @test y[2] ≈ 1.0
        @test y[3] ≈ 2.0
        @test y[4] ≈ 2.0
        
        # Test mul! with pre-allocated vector
        y2 = zeros(ComplexF64, 4)
        mul!(y2, op, x)
        @test y2 ≈ y
        
        # Test conversion to sparse
        sparse_mat = SciMLOperators.concretize(op)
        @test size(sparse_mat) == (4, 4)
        @test sparse_mat * x ≈ y
    end
    
    @testitem "BSR to CSC conversion" tags=[:BSR] begin
        using SparseArrays
        using LinearAlgebra
        
        # Test 1: Simple BSR matrix with identity blocks
        block_size = 2
        nrows = 3
        ncols = 3
        
        # Create identity-like blocks
        block_I = sparse([1.0+0im 0.0; 0.0 1.0])
        block_2I = sparse([2.0+0im 0.0; 0.0 2.0])
        block_3I = sparse([3.0+0im 0.0; 0.0 3.0])
        
        unique_blocks = [block_I, block_2I, block_3I]
        # Diagonal pattern: blocks at (1,1), (2,2), (3,3)
        block_rowval = [1, 2, 3]
        block_colptr = [1, 2, 3, 4]  # One block per column
        block_indices = [1, 2, 3]
        
        bsr = BlockSparseRowMatrix(block_size, nrows, ncols, unique_blocks, block_rowval, block_colptr, block_indices)
        
        # Convert using both functions
        csc1 = to_sparse(bsr)
        csc2 = bsr_to_csc(bsr)
        
        # Test that both functions produce the same result
        @test csc1 == csc2
        
        # Test dimensions
        @test size(csc1) == (6, 6)
        @test size(csc2) == (6, 6)
        
        # Test that diagonal elements are correct
        @test csc1[1, 1] == 1.0 + 0im
        @test csc1[2, 2] == 1.0 + 0im
        @test csc1[3, 3] == 2.0 + 0im
        @test csc1[4, 4] == 2.0 + 0im
        @test csc1[5, 5] == 3.0 + 0im
        @test csc1[6, 6] == 3.0 + 0im
        
        # Test off-diagonal elements are zero
        @test csc1[1, 2] == 0.0 + 0im
        @test csc1[1, 3] == 0.0 + 0im
        
        # Test 2: BSR with off-diagonal blocks and deduplication
        block_A = sparse([1.0+1.0im 2.0; 3.0 4.0+0im])
        block_B = sparse([5.0+0im 6.0; 7.0 8.0+1.0im])
        
        unique_blocks2 = [block_A, block_B]
        # Pattern: (1,1) and (1,2) use block_A (deduplicated), (2,1) uses block_B
        block_rowval2 = [1, 1, 2]
        block_colptr2 = [1, 3, 4]  # Column 1 has 2 blocks, column 2 has 1 block
        block_indices2 = [1, 2, 1]  # (1,1)->A, (2,1)->B, (1,2)->A (dedup!)
        
        bsr2 = BlockSparseRowMatrix(2, 2, 2, unique_blocks2, block_rowval2, block_colptr2, block_indices2)
        csc_converted = bsr_to_csc(bsr2)
        
        # Test dimensions
        @test size(csc_converted) == (4, 4)
        
        # Test that block (1,1) elements are correct
        @test csc_converted[1, 1] == 1.0 + 1.0im
        @test csc_converted[1, 2] == 2.0 + 0im
        @test csc_converted[2, 1] == 3.0 + 0im
        @test csc_converted[2, 2] == 4.0 + 0im
        
        # Test that block (2,1) elements are correct
        @test csc_converted[3, 1] == 5.0 + 0im
        @test csc_converted[3, 2] == 6.0 + 0im
        @test csc_converted[4, 1] == 7.0 + 0im
        @test csc_converted[4, 2] == 8.0 + 1.0im
        
        # Test that block (1,2) elements are correct (should be same as (1,1) due to dedup)
        @test csc_converted[1, 3] == 1.0 + 1.0im
        @test csc_converted[1, 4] == 2.0 + 0im
        @test csc_converted[2, 3] == 3.0 + 0im
        @test csc_converted[2, 4] == 4.0 + 0im
        
        # Test 3: Matrix-vector multiplication consistency
        x = randn(ComplexF64, 4)
        
        # Direct BSR multiplication (via element access)
        y_bsr = zeros(ComplexF64, 4)
        for j in 1:4
            for i in 1:4
                y_bsr[i] += bsr2[i, j] * x[j]
            end
        end
        
        # CSC multiplication
        y_csc = csc_converted * x
        
        # Should be identical
        @test y_bsr ≈ y_csc rtol=1e-14
        
        # Test 4: Large random sparse blocks
        block_size = 4
        nrows = 5
        ncols = 5
        
        # Create some random sparse blocks
        Random.seed!(42)
        block_rand1 = sprand(ComplexF64, block_size, block_size, 0.3)
        block_rand2 = sprand(ComplexF64, block_size, block_size, 0.3)
        
        unique_blocks3 = [block_rand1, block_rand2]
        # Place blocks at various positions with deduplication
        block_rowval3 = [1, 3, 5, 2, 4]
        block_colptr3 = [1, 2, 3, 5, 6, 6]  # Sparse column pattern
        block_indices3 = [1, 2, 1, 2, 1]  # Use blocks with deduplication
        
        bsr3 = BlockSparseRowMatrix(block_size, nrows, ncols, unique_blocks3, block_rowval3, block_colptr3, block_indices3)
        csc3 = bsr_to_csc(bsr3)
        
        # Test dimensions
        @test size(csc3) == (20, 20)
        
        # Test matrix-vector multiplication consistency
        x3 = randn(ComplexF64, 20)
        y_bsr3 = zeros(ComplexF64, 20)
        for j in 1:20
            for i in 1:20
                y_bsr3[i] += bsr3[i, j] * x3[j]
            end
        end
        y_csc3 = csc3 * x3
        
        @test y_bsr3 ≈ y_csc3 rtol=1e-12
        
        println("\n✓ BSR to CSC conversion tests passed:")
        println("  - Simple diagonal blocks: ✓")
        println("  - Off-diagonal blocks with deduplication: ✓")
        println("  - Matrix-vector multiplication consistency: ✓")
        println("  - Large random sparse blocks: ✓")
    end
    
    @testitem "M_Boson with BSR" tags=[:BSR] begin
        using SparseArrays
        # Use proper parameters from existing tests
        λ = 0.1450
        W = 0.6464
        kT = 0.7414
        N = 3
        tier = 2
        
        # System Hamiltonian
        Hsys = Qobj([
            0.6969 0.4364
            0.4364 0.3215
        ])
        
        # system-bath coupling operator
        Q = Qobj([
            0.1234 0.1357+0.2468im
            0.1357-0.2468im 0.5678
        ])
        
        # Create bosonic bath with Drude-Lorentz spectral density
        Bath = Boson_DrudeLorentz_Pade(Q, λ, W, kT, N)
        
        # Test standard CSC format
        M_csc = M_Boson(Hsys, tier, Bath, verbose=false)
        
        # Test BSR format
        M_bsr = M_Boson(Hsys, tier, Bath, EVEN, true, verbose=false)
        
        @test size(M_csc) == size(M_bsr)
        @test M_csc.N == M_bsr.N
        @test M_csc.sup_dim == M_bsr.sup_dim
        
        # Test that BSR operator behaves like CSC
        x = randn(ComplexF64, size(M_csc, 1))
        
        y_csc = M_csc.data * x
        y_bsr = M_bsr.data * x
        
        # They should give the same result
        @test y_csc ≈ y_bsr rtol=1e-10
        
        # Check that BSR is actually using the BSROperator
        @test M_bsr.data isa BSROperator
        
        # Check memory savings
        if M_bsr.data isa BSROperator
            bsr_mat = M_bsr.data.bsr
            println("\nBSR Memory Savings for Boson (tier=$tier):")
            println("  Total blocks: $(nnz_blocks(bsr_mat))")
            println("  Unique blocks: $(n_unique_blocks(bsr_mat))")
            println("  Savings: $(round((1 - memory_savings(bsr_mat)) * 100, digits=2))%")
            
            @test n_unique_blocks(bsr_mat) < nnz_blocks(bsr_mat)  # Should have deduplication
        end
    end
    
    @testitem "M_Fermion with BSR" tags=[:BSR] begin
        using SparseArrays
        # Use proper parameters from existing tests
        λ = 0.1450
        W = 0.6464
        kT = 0.7414
        μ = 0.8787
        N = 3
        tier = 2
        
        # System Hamiltonian
        Hsys = Qobj([
            0.6969 0.4364
            0.4364 0.3215
        ])
        
        # system-bath coupling operator (fermion annihilation operator)
        Q = Qobj([
            0.1234 0.1357+0.2468im
            0.1357-0.2468im 0.5678
        ])
        
        # Create fermionic bath with Lorentz spectral density
        Bath = Fermion_Lorentz_Pade(Q, λ, μ, W, kT, N)
        
        # Test standard CSC format
        M_csc = M_Fermion(Hsys, tier, Bath, verbose=false)
        
        # Test BSR format
        M_bsr = M_Fermion(Hsys, tier, Bath, EVEN, true, verbose=false)
        
        @test size(M_csc) == size(M_bsr)
        @test M_csc.N == M_bsr.N
        @test M_csc.sup_dim == M_bsr.sup_dim
        
        # Test that BSR operator behaves like CSC
        x = randn(ComplexF64, size(M_csc, 1))
        
        y_csc = M_csc.data * x
        y_bsr = M_bsr.data * x
        
        # They should give the same result
        @test y_csc ≈ y_bsr rtol=1e-10
        
        # Check that BSR is actually using the BSROperator
        @test M_bsr.data isa BSROperator
        
        # Check memory savings
        if M_bsr.data isa BSROperator
            bsr_mat = M_bsr.data.bsr
            println("\nBSR Memory Savings for Fermion (tier=$tier):")
            println("  Total blocks: $(nnz_blocks(bsr_mat))")
            println("  Unique blocks: $(n_unique_blocks(bsr_mat))")
            println("  Savings: $(round((1 - memory_savings(bsr_mat)) * 100, digits=2))%")
            
            @test n_unique_blocks(bsr_mat) < nnz_blocks(bsr_mat)  # Should have deduplication
        end
    end
    
    @testitem "M_Boson_Fermion with BSR" tags=[:BSR] begin
        using SparseArrays
        # Use proper parameters from existing tests
        λ = 0.1450
        W = 0.6464
        kT = 0.7414
        μ = 0.8787
        N = 2  # Smaller N for mixed bath to keep test fast
        Btier = 2
        Ftier = 2
        
        # System Hamiltonian
        Hsys = Qobj([
            0.6969 0.4364
            0.4364 0.3215
        ])
        
        # system-bath coupling operator
        Q = Qobj([
            0.1234 0.1357+0.2468im
            0.1357-0.2468im 0.5678
        ])
        
        # Create both types of baths with proper spectral densities
        Bbath = Boson_DrudeLorentz_Pade(Q, λ, W, kT, N)
        Fbath = Fermion_Lorentz_Pade(Q, λ, μ, W, kT, N)
        
        # Test standard CSC format
        M_csc = M_Boson_Fermion(Hsys, Btier, Ftier, Bbath, Fbath, verbose=false)
        
        # Test BSR format
        M_bsr = M_Boson_Fermion(Hsys, Btier, Ftier, Bbath, Fbath, EVEN, true, verbose=false)
        
        @test size(M_csc) == size(M_bsr)
        @test M_csc.N == M_bsr.N
        @test M_csc.sup_dim == M_bsr.sup_dim
        
        # Test that BSR operator behaves like CSC
        x = randn(ComplexF64, size(M_csc, 1))
        
        y_csc = M_csc.data * x
        y_bsr = M_bsr.data * x
        
        # They should give the same result
        @test y_csc ≈ y_bsr rtol=1e-10
        
        # Check that BSR is actually using the BSROperator
        @test M_bsr.data isa BSROperator
        
        # Check memory savings
        if M_bsr.data isa BSROperator
            bsr_mat = M_bsr.data.bsr
            println("\nBSR Memory Savings for Boson_Fermion (Btier=$Btier, Ftier=$Ftier):")
            println("  Total blocks: $(nnz_blocks(bsr_mat))")
            println("  Unique blocks: $(n_unique_blocks(bsr_mat))")
            println("  Savings: $(round((1 - memory_savings(bsr_mat)) * 100, digits=2))%")
            
            @test n_unique_blocks(bsr_mat) < nnz_blocks(bsr_mat)  # Should have deduplication
        end
    end

