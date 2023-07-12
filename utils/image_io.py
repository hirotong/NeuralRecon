import numpy as np

def load_pgm_image(image):
    with open(image, 'rb') as f:
        # file heads 
        head = f.readline()
        depth_shift = float(f.readline().split(b' ')[-1])
        width, height = map(lambda x: int(x), f.readline().split(b' '))
        end = int(f.readline())
        # assert end == 65535, "ERROR in pgm fileheader"
        # data is big endian in pgm file.
        dt = np.dtype('>H')
        data = np.frombuffer(f.read(width * height * 2), dtype=dt).reshape(height, width)
        result = data.astype(float) / depth_shift
        
    return result