# timings.py
import time,os

class Timings:
    def __init__(self, data = [], prec = 5):
        self.data = data
        self.prec = prec
        self.start_time = 0.
    
    def start(self):
        self.start_time = time.time()
    
    def add(self, label=""):
        self.data.append((label, time.time() - self.start_time))

    def list(self):
        print("Timings recorded")
        prec_str = "{0:." + str(self.prec) + "f}"
        for (l, t) in self.data:
            print(l + "\t\t" + prec_str.format(t) + " seconds")

