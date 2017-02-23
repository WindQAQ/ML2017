import sys
from PIL import Image

origin = Image.open(sys.argv[1])
modified = Image.open(sys.argv[2])

size, mode = origin.size, modified.mode
diff = Image.new(mode, size)

pix_o, pix_m, pix_d = origin.load(), modified.load(), diff.load()

for i in range(size[0]):
	for j in range(size[1]):
		pix_d[i, j] = pix_m[i, j] if pix_o[i, j] != pix_m[i, j] else (0, 0, 0, 0)

diff.save('ans_two.png')