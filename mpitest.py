import websky_stack_and_visualize as w
import websky_lensing_reconstruction as j
from pixell.reproject import healpix2map,thumbnails
from pixell import utils
from mpi4py import MPI
import numpy as np
import time

rad = np.deg2rad(6./60.)
res = np.deg2rad(0.5/60.)

def verify_stack(imap, ra, dec):
    coords = np.array([[dec[i], ra[i]] for i in range(len(ra))])
    thumbs = thumbnails(imap, coords, r=rad, res=res)
    return thumbs

def stack_random_all(imap, ra, dec):
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()

    if rank == 0:
        coords = np.array([[dec[i], ra[i]] for i in range(len(ra))])
        q, r = divmod(coords.size // 2, size)
        count = 2 * np.array([q + 1 if p < r else q for p in range(size)])
        disp = np.array([sum(count[:p]) for p in range(size)])
    else:
        coords = None
        count = np.zeros(size, dtype=np.int)
        disp = None
    
    comm.Bcast(count, root=0)
    coords_buf = np.zeros((count[rank] // 2, 2))

    comm.Scatterv([coords, count, disp, MPI.DOUBLE], coords_buf, root=0)
    #print(f"After scatter, process {rank} has coords {coords_buf}")

    thumbs = thumbnails(imap, coords_buf, r=rad, res=res)
    #print(f"After thumbs, process {rank} has data of size {thumbs.shape}")

    result = utils.allgatherv(thumbs, comm)
    #print(f"After all processes, process {rank} has data of size {result.shape}")

    return result

NUM = 101
comm = MPI.COMM_WORLD
ra = np.linspace(0.0, 3.0, num=NUM)
dec = np.linspace(-1.0, 1.0, num=NUM)

imap = j.almfile_to_map("websky/lensed_alm.fits")

if comm.Get_rank() == 0:
    a = time.time()
results = stack_random_all(imap, ra, dec)
if comm.Get_rank() == 0:
    b = time.time()
verify = verify_stack(imap, ra, dec)

if comm.Get_rank() == 0:
    r = np.sum(results, axis=0)
    v = np.sum(verify, axis=0)

    print(f"Max value in diff: {np.abs(r - v).max()}")
    print(f"Time taken for {NUM} calculations: {b-a: 0.5f} seconds")

