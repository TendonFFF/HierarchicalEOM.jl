# @testitem "Block Sparse Row (BSR) Matrix" begin
    @testitem "BlockSparseRowMatrix basics" begin
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
    
    @testitem "BSRBuilder" begin
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
    
    @testitem "BSROperator" begin
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
    
    @testitem "M_Boson with BSR" begin
        using SparseArrays
        # Create a simple 2-level system
        Hsys = sigmaz()
        
        # Create a bosonic bath with small parameters for testing
        Bath = BosonBath(Hsys, 0.1, 1.0, 0.5, 3)
        
        tier = 2
        
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
            println("\nBSR Memory Savings for tier=$tier:")
            println("  Total blocks: $(nnz_blocks(bsr_mat))")
            println("  Unique blocks: $(n_unique_blocks(bsr_mat))")
            println("  Savings: $(round((1 - memory_savings(bsr_mat)) * 100, digits=2))%")
            
            @test n_unique_blocks(bsr_mat) < nnz_blocks(bsr_mat)  # Should have deduplication
        end
    end
# end
