import av
import PIL

# from matplotlib.pyplot import imshow
import numpy as np

import skimage.io
from skimage.transform import resize, pyramid_reduce


container = av.open('20151113_230303_00_Video.avi')

br_count = 0
for frame in container.decode(video=0):
    print("frame index ========================", frame.index, type(frame))

    img_pil = frame.to_image()
    width, height = img_pil.size  # width, height for PIL
    print("width, height", width, height)

    # converting PIL (<class 'PIL.Image.Image'>) to <class 'numpy.ndarray'>
    print("type(img_pil)", type(img_pil))
    img_pil_arr = np.asarray(img_pil)
    print("type(img_pil_arr)", type(img_pil_arr))

    h, w, c = img_pil_arr.shape  # h, w, c for skimage
    print("img_pil_arr.shape (h, w, c)", h, w, c, img_pil_arr.shape)  # this is exact same as when you read a jpg image using skimage.io.imread (h, w, c)

    #
    # 1 crop and resize using PIL
    img_central_pil = img_pil.crop((240, 0, width-240, height))  # (x0, y0, x1, y1)
    print("img_central_pil", img_central_pil.size)
    mywidth = 320
    wpercent = (mywidth / float(img_central_pil.size[0]))
    hsize = int((float(img_central_pil.size[1]) * float(wpercent)))
    img_down_pil = img_central_pil.resize((mywidth, hsize), PIL.Image.ANTIALIAS)
    # img_down_pil.save('pil_frame-%04d_240x320.jpg' % frame.index)

    # 2 crop and resize using skimage.io
    img_central_sk = img_pil_arr[0:h, 240:w-240]  # x0:x1,y0:y1
    img_down_sk = pyramid_reduce(img_central_sk, downscale=4.5)
    skimage.io.imsave('sk_frame-%04d_240x320.jpg' % frame.index, img_down_sk)


    # original image saved
    # img_pil.show()
    img_pil.save('frame-%04d.jpg' % frame.index)

    br_count += 1
    if br_count > 5:
        break
