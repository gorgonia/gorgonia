import glob
from subprocess import call
from os.path import basename, splitext

compute = [30, 32, 35, 37, 50, 52, 53, 60, 61, 62]
def main():
	src = glob.glob("src/*.cu")
	cmd = ['nvcc', '-o=', '-arch=', '-ptx', '-Xptxas', '-allow-expensive-optimizations']
	# slow
	cmdslow = cmd[:]
	cmdslow.extend(['-fmad=false', '-ftz=false', '-prec-div=true', '-prec-sqrt=true', "INPUTFILE"])

	for f in src:
		name, ext = splitext(basename(f))
		for cc in compute:
			cmdslow[1] = '-o="target/' + name + '_cc' + str(cc) + '.ptx"'
			cmdslow[-1] = f
			cmdslow[2] = '-arch=compute_'+str(cc)
			print(cmdslow)
			call(cmdslow)

	#fast
	cmdfast = cmd[:]
	cmdfast.extend(['-fmad=false', '-use_fast_math', "INPUTFILE"])

if __name__ == '__main__':
	main()