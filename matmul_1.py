import torch
import os
import cutlass.cute as cute
import cutlass
import cuda.bindings.driver as cuda
from cutlass.cute.runtime import from_dlpack
from functools import partial
from benchmark import benchmark

from typing import Optional, Type, Tuple, Union

import cutlass.utils.hopper_helpers as sm90_utils

os.environ["CUTE_DSL_ARCH"] = "sm_90a"

#### Gotchas:
# 1. must use 1 warpgroup per BLOCK_M / MMA_M, results are slightly incorrect otherwise, not sure of why this is
# 2. warpgroup 0 must be a consumer warpgroup, because of async pipeline implementation

# this kernel has a heisenbug, I think where producer/consumer get out of sync because barriers are using an arrival count of 1
# printing at the end of each persistent kernel iteration seems to fix it. The async pipeline implementation requires an arrival
class GemmKernel:
    def __init__(self):
        self.num_consumer_warpgroups = 2
        self.num_producer_warpgroups = 1
        self.BM = 128
        self.BN = 256
        self.BK = 64
        self.PIPELINE_STAGES = 4

        self.accumulator_dtype = cutlass.Float32
    
    @cute.jit
    def __call__(self, A, B, C, stream):
        
        ####################################
        ### create shared memory layouts ###
        ####################################

        # determines how shared memory should be swizzled, given
        # its width, dtype, and layout
        # SmemLayoutAtomKind.K_SW128, 128B swizzle
        A_smem_layout_atom_kind = sm90_utils.get_smem_layout_atom(
            cutlass.utils.layout.LayoutEnum.ROW_MAJOR,
            A.element_type,
            self.BK, # size of the shared memory major mode dimension
        )

        # produces an actual smem layout atom, i think this is sort of an 'atomic'
        # chunk of smem, given the swizzling layout determined above, and a dtype
        # in this case,  S<3,4,3> o 0 o (8,64):(64,1)
        # this is 8 rows of 128 bytes each, the smallest chunk of smem that can accomodate
        # this swizzling pattern
        a_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            A_smem_layout_atom_kind,
            A.element_type
        )

        # tile this smem layout atom across the full amount of shared memory that we want
        # gives us S<3,4,3> o 0 o ((8,16),(64,1),(1,4)):((64,512),(1,0),(0,8192))
        # 1st mode is BM, second mode is BK, third mode is pipeline stages
        a_smem_layout_staged = cute.tile_to_shape(
            a_smem_layout_atom,
            (self.BM, self.BK, self.PIPELINE_STAGES),
            order=(0,1,2)
        )

        # SmemLayoutAtomKind.K_SW128, 128B swizzle
        B_smem_layout_atom_kind = sm90_utils.get_smem_layout_atom(
            cutlass.utils.layout.LayoutEnum.ROW_MAJOR,
            B._dtype,
            self.BK, # size of the shared memory major mode dimension, A/B are both K major
        )

        # (8,64):(64,1)
        b_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            B_smem_layout_atom_kind,
            B._dtype
        )

        # ((8,32),(64,1),(1,4)):((64,512),(1,0),(0,16384))
        b_smem_layout_staged = cute.tile_to_shape(
            b_smem_layout_atom,
            (self.BN, self.BK, self.PIPELINE_STAGES),
            order=(0,1,2)
        )

        @cute.struct
        class SharedStorage:
            mainloop_pipeline_array_ptr: cute.struct.MemRange[
                cutlass.Int64, self.PIPELINE_STAGES * 2 # two barrier per pipeline stage
            ]
            sa: cute.struct.Align[
                cute.struct.MemRange[
                    A.element_type, cute.cosize(a_smem_layout_staged)
                ],
                1024, # 1kb alignment
            ]
            sb: cute.struct.Align[
                cute.struct.MemRange[
                    B.element_type, cute.cosize(b_smem_layout_staged)
                ],
                1024, # 1kb alignment
            ]
        self.shared_storage = SharedStorage

        ###############################
        ### create tiled mma object ###
        ###############################
        # Thr Layout VMNK: (128,2,1,1):(1,128,0,0)
        # Permutation MNK: (_,_,_)
        # MMA Atom
        # ThrID:           128:1
        # Shape MNK:       (64,256,16)
        # TV Layout A:     (128,(64,16)):(0,(1,64))     # (threads, (mma_m, mma_k))
        # TV Layout B:     (128,(256,16)):(0,(1,256))   # (threads, (mma_n, mma_k))
        # TV Layout C:     ((4,8,4),(2,2,32)):((128,1,16),(64,8,512))
        tiled_mma = sm90_utils.make_trivial_tiled_mma(
            A.element_type,
            B.element_type,
            cute.nvgpu.warpgroup.OperandMajorMode.K, # A is K major
            cute.nvgpu.warpgroup.OperandMajorMode.K, # B is K major
            self.accumulator_dtype,
            (self.num_consumer_warpgroups, 1, 1), # atom layout mnk, number of warpgroups per dimension
            tiler_mn=(64, self.BN) # i think this is the output shape we want from a single mma instruction?
        )

        ################################
        # create tma atoms and tensors #
        ################################
        # tma_atom_a: Copy Atom
        # ThrID:         1:0
        # TV Layout Src: (1,8192):(0,1)
        # TV Layout Dst: (1,8192):(0,1)
        # Value type:    bf16
        # tma_tensor_a: tensor<(0,0) o (8192,8192):(1@1,1@0)>
        tma_atom_a, tma_tensor_a = cute.nvgpu.cpasync.make_tma_tile_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(),
            A,
            cute.slice_(a_smem_layout_staged, (None, None, 0)), # slice out the pipeline stages
            (self.BM, self.BK),
            num_multicast=1
        )

        tma_atom_b, tma_tensor_b = cute.nvgpu.cpasync.make_tma_tile_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(),
            B,
            cute.slice_(b_smem_layout_staged, (None, None, 0)), # slice out the pipeline stages
            (self.BN, self.BK),
            num_multicast=1
        )

        M = A.layout.shape[0]
        N = B.layout.shape[0]
        K = A.layout.shape[1]
        
        # the cutlass.const_expr is just for dev, in real world you'd want to have these checks
        # at the pytorch level, since these are akin to static_asserts
        if cutlass.const_expr(M % self.BM != 0 or N % self.BN != 0 or K % self.BK != 0):
            raise ValueError(f"M, N, K must be divisible by BM, BN, BK, got {M}, {N}, {K}")
        if cutlass.const_expr(A.layout.shape[1] != B.layout.shape[1]):
            raise ValueError(f"A.K != B.K, got {A.layout.shape[1]}, {B.layout.shape[1]}")

        # check strides
        if cutlass.const_expr(A.layout.stride[1] != 1):
            raise ValueError(f"A.stride[1] != 1, got {A.layout.stride[1]}")
        if cutlass.const_expr(B.layout.stride[1] != 1):
            raise ValueError(f"B.stride[0] != 1, got {B.layout.stride[0]}")

        tile_sched_params, grid = self._compute_grid(
            M,
            N,
            self.BM,
            self.BN
        )

        num_threads = 128 * (self.num_consumer_warpgroups + self.num_producer_warpgroups)
        block_dim = (num_threads, 1, 1)

        cute.printf("grid: {}", grid)
        cute.printf("block_dim: {}", block_dim)
        
        # since this kernel uses tma for accessing A and B, it only recieves the tma atoms and tensors
        self.kernel(
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            C,
            tiled_mma,
            a_smem_layout_staged,
            b_smem_layout_staged,
            tile_sched_params,
        ).launch(
            grid=grid,
            block=block_dim,
            cluster=(1,1,1),
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
        )

    @cute.kernel
    def kernel(self,
               tma_atom_a : cute.CopyAtom,
               tma_coord_A_mk : cute.Tensor,
               tma_atom_b : cute.CopyAtom,
               tma_coord_B_nk : cute.Tensor,
               gC : cute.Tensor,
               tiled_mma : cute.TiledMma,
               a_smem_layout_staged : cute.ComposedLayout,
               b_smem_layout_staged : cute.ComposedLayout,
               tile_sched_params : cutlass.utils.PersistentTileSchedulerParams,
    ):
        
        # setup
        bidx, bidy, bidz = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        grid_dim_x, _, _ = cute.arch.grid_dim()
        
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        warp_idx_in_warpgroup = cute.arch.warp_idx() % 4
        warp_idx_in_warpgroup = cute.arch.make_warp_uniform(warp_idx_in_warpgroup)
        warp_group_idx = cute.arch.make_warp_uniform(
            cute.arch.warp_idx() // 4
        )

        is_consumer = warp_group_idx == 0 or warp_group_idx == 1
        is_producer = not is_consumer
        
        if warp_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_b)

        # allocate shared memory #
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        # get pointer to pipeline barriers in shared memory
        mainloop_pipeline_array_ptr = storage.mainloop_pipeline_array_ptr.data_ptr()
        
        
        mainloop_pipeline_producer_group = cutlass.utils.CooperativeGroup(cutlass.utils.Agent.Thread)
        mainloop_pipeline_consumer_group = cutlass.utils.CooperativeGroup(
            cutlass.utils.Agent.Thread, size = 1
        )
        
        # size of one pipeline stage BM * BK + BN * BK
        tma_copy_bytes = (self.BM * self.BK + self.BN * self.BK) * 2
        cta_layout_mnk = cute.make_layout((1, 1, 1)) #  no tma multicast
        
        mainloop_pipeline = cutlass.utils.PipelineTmaAsync.create(
            barrier_storage=mainloop_pipeline_array_ptr,
            num_stages=self.PIPELINE_STAGES,
            producer_group=mainloop_pipeline_producer_group,
            consumer_group=mainloop_pipeline_consumer_group,
            tx_count=tma_copy_bytes,
            cta_layout_vmnk=cta_layout_mnk
        )

        #################################
        # get pointers to smem tensors ##
        #################################
        # smem_layout_staged.outer is the plain layout without swizzling
        # for a in this case it is ((8,16),(64,1),(1,4)):((64,512),(1,0),(0,8192))
        # smem_layout_staged.inner is the swizzle function, S<3,4,3>
        sa = storage.sa.get_tensor(a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner)
        sb = storage.sb.get_tensor(b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner)
        
        ####################
        ## producer setup ##
        ####################

        #####################################################
        # partition tma coordinates into block tiles ########
        #####################################################
        # tma_coord_A_mk: (M, K) : ((0,1), (1, 0)) the stride has the effect of mapping tensor coordinate (i,j) to tma coordinate (j,i), why?
        # after tiling by (BM, BK) we get:
        # tma_coord_A_tiled: (BM, BK, M / BM, K / BK)
        tma_coord_A_tiled = cute.local_tile(
            tma_coord_A_mk, # (M, K)
            (self.BM, self.BK),
            (None, None) # keep both m,k modes m will be indexed in outer loop of persistent kernel, k in inner loop
        )

        tma_coord_B_tiled = cute.local_tile(
            tma_coord_B_nk,
            (self.BN, self.BK),
            (None, None)
        )

        ###########################################################################################
        # use cpasync.tma_partition to align views of smem and gmem so that cute.copy can be used #
        ###########################################################################################
        # tma_partition expects to find coordinates of global memory tiles in the 0th mode of tma_coord_A_tiled_tma
        #    (BM, BK), num_block_m, num_block_k
        # and corresponding tiles of smem in the 0th mode of sa_for_tma_partition
        #    (BM, BK), pipeline stage
        sa_for_tma_partition = cute.group_modes(sa, 0, 2)
        tma_coord_A_tiled_tma = cute.group_modes(tma_coord_A_tiled, 0, 2)
        a_cta_layout = cute.make_layout((1,))
        a_cta_crd = 0
        tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
            tma_atom_a,
            a_cta_crd,
            a_cta_layout,
            sa_for_tma_partition, # (SMEM_TILE_A, STAGE)
            tma_coord_A_tiled_tma # (SMEM_TILE_A, NUM_BLOCKS_M, NUM_BLOCKS_K)
        )
        # tAsA:     (SMEM_TILE_A, NUM_STAGES)
        # tAgA: (MMA_A, NUM_BLOCKS_M, NUM_BLOCKS_K)

        sb_for_tma_partition = cute.group_modes(sb, 0, 2)
        tma_coord_B_tiled_tma = cute.group_modes(tma_coord_B_tiled, 0, 2)
        b_cta_layout = cute.make_layout((1,))
        b_cta_crd = 0
        tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
            tma_atom_b,
            b_cta_crd,
            b_cta_layout,
            sb_for_tma_partition, # (SMEM_TILE_B, STAGE)
            tma_coord_B_tiled_tma # (SMEM_TILE_B, NUM_BLOCKS_N, NUM_BLOCKS_K)
        )
        # tBsB: (SMEM_TILE_B, NUM_STAGES)
        # tBgB: (MMA_B, NUM_BLOCKS_N, NUM_BLOCKS_K)
        num_tiles_k = cute.size(tAgA, mode=[2])

        ########################
        ## end producer setup ##
        ########################

        ####################
        ## consumer setup ##
        ####################

        C_tiled = cute.local_tile(
            gC,
            (self.BM, self.BN),
            (None, None)
        ) # (128,256,1,1):(256,1,0,0)
        
        # get the chunk of the tiled mma that this warp group is responsible for
        thr_mma = tiled_mma.get_slice(tidx % 256)
        tCsA = thr_mma.partition_A(sa) # TODO are these correctly pointing to different tiles of shared memory?
        tCsB = thr_mma.partition_B(sb)
        tCgC = thr_mma.partition_C(C_tiled)
        # C_tiled : (BM, BN, M / BM, N / BN)
        # tCgC : (thread_idx within warpgroup, BM / MMA_M, BN / MMA_N, M / BM, N / BN)
        tCrA = tiled_mma.make_fragment_A(tCsA)
        tCrB = tiled_mma.make_fragment_B(tCsB)

        # if bidz == 0 and tidx == 0:
        #     cute.printf("--------------------------------")
        #     cute.printf("thr_mma: "); cute.printf(str(thr_mma)) # shouldnt second mode be 2?
        #     cute.printf("tCsA: "); cute.printf(str(tCsA.layout)) # ((64,16),2,4,(1,4)):((64,1),4096,16,(0,8192)) # ((MMA_M, MMA_K), BM / MMA_M, BK / MMA_K, pipeline_stages)
        #     cute.printf("tCsB: "); cute.printf(str(tCsB.layout)) # ((256,16),1,4,(1,4)):((64,1),0,16,(0,16384)) # ((MMA_N, MMA_K), BN / MMA_N, BK / MMA_K, pipeline_stages)
        #     cute.printf("tCrA: "); cute.printf(str(tCrA.layout)) # (1,1,4,(1,4)):(0,0,2,(0,1024))
        #     cute.printf("tCrB: "); cute.printf(str(tCrB.layout)) # (1,1,4,(1,4)):(0,0,2,(0,2048))
        #     cute.printf("C_tiled: "); cute.printf(str(C_tiled.layout))
        #     cute.printf("tCgC: "); cute.printf(str(tCgC.layout)) # ((2,2,32),1,1,64,32):((1,65536,8),0,0,1048576,256)
        #     cute.printf("--------------------------------")
        
        
        consumer_state = cutlass.utils.make_pipeline_state(cutlass.utils.PipelineUserType.Consumer, self.PIPELINE_STAGES)
        producer_state = cutlass.utils.make_pipeline_state(cutlass.utils.PipelineUserType.Producer, self.PIPELINE_STAGES)
        
        cute.arch.sync_threads()

        if is_producer:
            if warp_idx_in_warpgroup == 0:
                tile_sched = cutlass.utils.StaticPersistentTileScheduler.create(
                    tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
                )
                
                work_tile = tile_sched.initial_work_tile_info()

                while work_tile.is_valid_tile:

                    current_tile_coord = work_tile.tile_idx
                    
                    tile_m = current_tile_coord[0]
                    tile_n = current_tile_coord[1]
                    coord_k = current_tile_coord[2]

                    # if (tidx == 0 or tidx == 256):
                    #     cute.printf("producer {} started {}", bidz, current_tile_coord)

                    for tile_k in cutlass.range_dynamic(num_tiles_k):                            
                        # block until this stage is ready to be written to
                        # call empty_barrier.wait(producer_state.index, producer_state.phase)
                        #   block until the phase of empty_buffer[index] != producer_state.phase (cute.arch.mbarrier_wait)
                        # call full_barrier.arrive(producer_state.index)
                        #   since this barrier is of type tma op, it calls cute.arch.mbarrier_init_tx_bytes
                        #   which tells the barrier to expect tx_count_bytes bytes to be written. Since the
                        #   barrier is of type tma op, we never arrive at it, the arrival count is not used
                        
                        # if bidz == 68 and (tidx == 0 or tidx == 256):
                        #     cute.printf("producer acquire, producer tile_m: {}, producer tile_n: {}, producer tile_k: {}, producer_state.index: {}, producer_state.phase: {}", tile_m, tile_n, tile_k, producer_state.index, producer_state.phase)
                        
                        mainloop_pipeline.producer_acquire(producer_state)

                        # if bidz == 68 and (tidx == 0 or tidx == 256):
                        #     cute.printf("producer acquire done")

                        # get the k slice of A and B that we are going to read from gmem
                        # and the corresponding stage of the shared memory buffer that are going to be written to
                        # these both come from cpasync.tma_partition
                        tAgA_k_index = (
                            None, # MMA
                            tile_m,
                            tile_k
                            )
                        
                        tAsA_stage_index = (
                            None, # smem
                            producer_state.index # stage index
                        )
                        tBgB_k_index = (
                            None,
                            tile_n,
                            tile_k
                        )
                        tBsB_stage_index = (
                            None,
                            producer_state.index
                        )

                        tAgA_k = tAgA[tAgA_k_index]
                        tAsA_stage = tAsA[tAsA_stage_index]
                        tBgB_k = tBgB[tBgB_k_index]
                        tBsB_stage = tBsB[tBsB_stage_index]

                        cute.copy(
                            tma_atom_a,
                            tAgA_k,
                            tAsA_stage,
                            tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                                producer_state
                            ),
                            mcast_mask=0
                        )
                        cute.copy(
                            tma_atom_b,
                            tBgB_k,
                            tBsB_stage,
                            tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                                producer_state
                            ),
                            mcast_mask=0
                        )

                        # if bidz == 68 and (tidx == 0 or tidx == 256):
                        #     cute.printf("producer commit, producer tile_m: {}, producer tile_n: {}, producer tile_k: {}, producer_state.index: {}, producer_state.phase: {}", tile_m, tile_n, tile_k, producer_state.index, producer_state.phase)

                        mainloop_pipeline.producer_commit(producer_state)

                        # if bidz == 68 and (tidx == 0 or tidx == 256):
                        #     cute.printf("producer commit done")

                        
                        producer_state.advance()


                    if (tidx == 0 or tidx == 256):
                        cute.printf("producer {} finished {}", bidz, current_tile_coord)
                    
                    # producer_state.count is the number of tiles that have been processes
                    # producer_state.index is the current pipeline stage
                    # this should reset count, but not index
                    # this may not be necessary, since we do not use count in the loop above
                    # (it could be used in place of tile_k)
                    producer_state.reset_count()
                    tile_sched.advance_to_next_work()
                    work_tile = tile_sched.get_current_work()

                # wait until all buffers are empty
                # mainloop_pipeline.producer_tail(producer_state)

                # if (tidx == 0 or tidx == 256):
                #     cute.printf("PRODUCER EXIT")

                # if bidz == 0 and tidx == 0:
                #     cute.printf("--------------------------------")
                #     cute.printf("tAgA_k: {}", tAgA_k.layout)
                #     cute.printf("tAsA_stage: {}", tAsA_stage.layout)
                #     cute.printf("tBgB_k: {}", tBgB_k.layout)
                #     cute.printf("tBsB_stage: {}", tBsB_stage.layout)
                #     cute.printf("--------------------------------")
        

        
        if is_consumer:

            tile_sched = cutlass.utils.StaticPersistentTileScheduler.create(
                    tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
                )

            work_tile = tile_sched.initial_work_tile_info()
            mma_k_per_block_k = cute.size(tCrA, mode=[2])

            while work_tile.is_valid_tile:

                accumulators = cute.make_fragment(tCgC[None, None, None, 0, 0].shape, self.accumulator_dtype)
                tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)

                for i in range(cute.size(accumulators)):
                    accumulators[i] = 0.0

                current_tile_coord = work_tile.tile_idx
                tile_m = current_tile_coord[0]
                tile_n = current_tile_coord[1]
                coord_k = current_tile_coord[2]

                # if (tidx == 0 or tidx == 256):
                #     cute.printf("consumer {} started {}", bidz, current_tile_coord)

                for tile_k in cutlass.range_dynamic(num_tiles_k):

                    # if bidz == 68 and (tidx == 0 or tidx == 256):
                    #     cute.printf("consumer wait, consumer tile_m: {}, consumer tile_n: {}, consumer tile_k: {}, consumer_state.index: {}, consumer_state.phase: {}", tile_m, tile_n, tile_k, consumer_state.index, consumer_state.phase)

                    
                    # wait at the full barrier, until the expected number of bytes have been written
                    mainloop_pipeline.consumer_wait(consumer_state)

                    # if bidz == 68 and (tidx == 0 or tidx == 256):
                    #     cute.printf("consumer wait done")
                        
                    cute.nvgpu.warpgroup.fence()
                    for mma_k in range(mma_k_per_block_k):

                        k_block_coord = (
                            None,
                            None,
                            mma_k,
                            consumer_state.index
                        )

                        tCrA_1phase = tCrA[k_block_coord]
                        tCrB_1phase = tCrB[k_block_coord]

                        cute.gemm(
                            tiled_mma,
                            accumulators,
                            tCrA_1phase,
                            tCrB_1phase,
                            accumulators,
                        )

                    cute.nvgpu.warpgroup.commit_group()
                    cute.nvgpu.warpgroup.wait_group(0)
                    
                    # the signalling thread calls cute.arch.mbarrier_arrive of the empty barrier
                    # since the arrival count of the empty barrier is 1, this is all we need to proceed
                    
                    # if bidz == 68 and (tidx == 0 or tidx == 256):
                    #     cute.printf("consumer release, consumer tile_m: {}, consumer tile_n: {}, consumer tile_k: {}, consumer_state.index: {}, consumer_state.phase: {}", tile_m, tile_n, tile_k, consumer_state.index, consumer_state.phase)
                    
                    mainloop_pipeline.consumer_release(consumer_state)

                    # if bidz == 68 and (tidx == 0 or tidx == 256):
                    #     cute.printf("consumer release done")
                    
                    consumer_state.advance()

                store_copy = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.accumulator_dtype)
                cute.copy(store_copy, accumulators, tCgC[None, None, None, tile_m, tile_n])
                
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

                if (tidx == 0 or tidx == 256):
                    cute.printf("consumer {} finished {}", bidz, current_tile_coord)
            
            # if (tidx == 0 or tidx == 256):
            #     cute.printf("CONSUMER EXIT")
            


    @staticmethod
    def _compute_grid(
        M : int,
        N : int,
        BM : int,
        BN : int
    ):
        num_ctas_mnl = (M // BM, N // BN, 1)
        cluster_shape_mnl = (1, 1, 1)
        max_active_clusters = cutlass.const_expr(132)
        
        tile_sched_params = cutlass.utils.PersistentTileSchedulerParams(
            num_ctas_mnl, cluster_shape_mnl
        )
        grid = cutlass.utils.StaticPersistentTileScheduler.get_grid_shape(
            tile_sched_params, max_active_clusters
        )
        return tile_sched_params, grid
        


if __name__ == "__main__":
    M,N,K = 4096, 4096, 4096
    # M,N,K = 1024, 1024, 1024
    torch.manual_seed(3)
    A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    C = torch.zeros(M, N, device="cuda", dtype=torch.float32)

    gA = from_dlpack(A, assumed_align=16)
    gB = from_dlpack(B, assumed_align=16)
    gC = from_dlpack(C, assumed_align=16)

    gemm = GemmKernel()

    torch_stream = torch.cuda.Stream()
    stream = cuda.CUstream(torch_stream.cuda_stream)
    compiled_gemm = cute.compile(gemm, gA, gB, gC, stream)

    torch_stream.synchronize()

    for i in range(1):
        print(f"iteration {i}")
        compiled_gemm(gA, gB, gC, stream)

    C = C.to(torch.bfloat16)

    C_ref = torch.matmul(A, B.t())

    diff = C - C_ref
    incorrect_indices_row, incorrect_indices_col = torch.where(diff != 0)
    unique_incorrect_indices = torch.unique(incorrect_indices_row)
    print(unique_incorrect_indices)
    assert torch.allclose(C, C_ref, atol=1e-3)
    
    # print("test passed")
    
    
    

