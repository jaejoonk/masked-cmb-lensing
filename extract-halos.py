import websky_stack_and_visualize as josh_websky
import time

print("Running extraction for 1e14 solar mass halos:")
t1 = time.time()
josh_websky.catalog_to_coords(filename="halos.pksc", mass_cutoff=1.0,
                              output_to_file=True, output_file="halos-1e14.txt")
t2 = time.time()
print("Done. Time elapsed: %0.5f seconds." % (t2 - t1))
print("")
print("Running extraction for 4e14 solar mass halos:")

t1 = time.time()
josh_websky.catalog_to_coords(filename="halos.pksc", mass_cutoff=4.0,
                              output_to_file=True, output_file="halos-4e14.txt")
t2 = time.time()
print("Done. Time elapsed: %0.5f seconds." % (t2 - t1))
