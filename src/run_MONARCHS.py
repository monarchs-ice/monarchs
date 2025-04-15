from monarchs.core.driver import monarchs

if __name__ == "__main__":

    try:
        from mpi4py import MPI
        rank = MPI.COMM_WORLD.Get_rank()
        print('MPI rank = ', rank)
    except:
        print("mpi4py not found, running in serial mode")
        use_mpi = False
        rank = 0
    if rank == 0:
        grid = monarchs()
    else:
        print('Rank {} not running monarchs'.format(rank))
