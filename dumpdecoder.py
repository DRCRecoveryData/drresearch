import LDPC_decoder as LDPCdecoder
import scipy.io as scio
import numpy as np
import multiprocessing
import sys

## LDPC Decoder for NAND Flash dump files


print("Usage: dumpdecoder.py <dumpfile.dump> <ldpcparameter.npy> <geometry.case> <output.dump>")


inputdump=sys.argv[1]
ldpcparam=sys.argv[2]
pattern=sys.argv[3]
outputdump=sys.argv[4]

#pagesize = 4000
#dataeccpos = [0, 1500]
#datasize=1024
#eccsize=476

pagesize = 20
dataeccpos = [0]
datasize=8
eccsize=8

## Noise
EbN = 3
print("EbN: "+str(EbN))

SNR_lin = 10**(EbN/10)
print("SNR_lin: "+str(SNR_lin))
No = 1.0/SNR_lin
print("No: "+str(No))
sigma = np.sqrt(No/2)
print("sigma: "+str(sigma))


H=np.load(ldpcparam)

with open(inputdump,"rb") as dump:
	with open(outputdump,"wb") as output:
		while dump:
			page = bytearray(dump.read(pagesize))
			for pos in dataeccpos:
				x = np.unpackbits(np.frombuffer(page[pos:pos+datasize+eccsize],dtype=np.uint8),axis=None,bitorder='little').astype(np.float32)
				#print("x: "+str(x))
				r = 2*x-1
				#print("r: "+str(r))
				#print("H: "+str(H))
				decoder = LDPCdecoder.decoder(H)
				#print("r: "+str(r))
				#print("size of r: "+str(r.shape))
				decoder.setInputMSA(r, sigma)
				
				# Get Hard-Bits
				w0 = r
				w0[w0 >= 0] = 1
				w0[w0 < 0] = 0
				w0 = np.array(w0, dtype = int)
				#ErrorUncoded = np.sum(w0 != x)
				#print("Amount of Bit Errors (uncoded) : %d " % ErrorUncoded)
				
				#MSA algorithm
				for n in range(0,200):
		    
					decoded, y = decoder.iterateMinimumSumAlgorithm()
					ErrorSPA = np.sum(y != x)
		
					print("Amount of Bit Errors (SPA) : %d " % ErrorSPA)
					if(decoded):
						break
		
				ErrorSPA = np.sum(y != x)
				print("Iterations:  %d  |  Amount of Bit Errors (SPA) : %d " % (n, ErrorSPA))
				#print("he")
				#print(y)
				#print("Original:")
				#print(u)
				page[pos:pos+datasize+eccsize]=np.packbits(decoded,axis=None,bitorder='little').tobytes()
			output.write(page)
		
