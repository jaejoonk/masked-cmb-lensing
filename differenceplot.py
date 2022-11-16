from pixell import enmap
from orphics import io
import numpy as np

imap = enmap.read_map("inpainted_null_map_beam_conv_6000.fits")
umap = enmap.read_map("uninpainted_null_map_beam_conv_6000.fits")

diff = imap-umap
io.hplot(diff, "inpaint-minus-uninpaint-halfscale", downgrade=2, colorbar=True)
print(f"Minimum value: {np.min(diff)} | Maximum value: {np.max(diff)}")
