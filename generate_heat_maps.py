import os
import skimage.io
from skimage.transform import resize, rescale, pyramid_reduce
import matplotlib.pyplot as plt
import numpy as np
from random import shuffle


def get_heat_map(index_array, left, right):
    # index_array (240/8, 320/8, 2), left [lx, ly], right [lr, ly]
    left_array = np.zeros(index_array.shape)

    if left[0] is None:
        heat_map_0 = left_array[:,:,0]
    else:
        left_array[:, :, 0] = left[1]
        left_array[:, :, 1] = left[0]
        heat_map_0 = np.exp(-np.sqrt(np.sum(np.square(index_array - left_array), axis=2)))
        heat_map_0 /= np.max(heat_map_0)

    right_array = np.zeros(index_array.shape)

    if right[0] is None:
        heat_map_1 = right_array[:,:,0]
    else:
        right_array[:, :, 0] = right[1]
        right_array[:, :, 1] = right[0]
        heat_map_1 = np.exp(-np.sqrt(np.sum(np.square(index_array - right_array), axis=2)))
        heat_map_1 /= np.max(heat_map_1)

    #print left, right, heat_map_0.shape, heat_map_1.shape, np.max(heat_map_0), np.max(heat_map_1)

    heat_map = heat_map_0+heat_map_1

    if np.sum(heat_map)>0:
        heat_map /= np.max(heat_map)

    #plt.imshow(heat_map)
    #plt.show()
    return heat_map

def save_dataset():
    labels_root = "/s/red/a/nobackup/cwc/palm_recognition/palm_labels/"
    labels = np.vstack([np.load(os.path.join(labels_root,numpy_file)) for numpy_file in os.listdir(labels_root)])
    print labels.shape

    index_array = np.zeros((240/8, 320/8, 2))

    for i in range(index_array.shape[0]):
        for j in range(index_array.shape[1]):
            index_array[i][j] = [i, j]  # height (y), width (x) => index_array[:,:,0] = y pixel coordinate and index_array[:,:,1] = x

    X = []
    y = []
    count = 1
    for file_name, l_x, l_y, r_x, r_y in labels:

        left = [l/8  if l is not None else l for l in [l_x, l_y]]
        right = [r/8 if r is not None else r for r in [r_x, r_y]]

        print count, file_name, left, right

        heat_map = get_heat_map(index_array, left, right)

        image = skimage.io.imread(file_name, True)

        X.append(image)
        y.append(heat_map)
        count += 1


    X = np.stack(X)
    y = np.stack(y)

    X = X[:,:,:,np.newaxis]

    indices = range(X.shape[0])
    shuffle(indices)
    print indices

    X_train = X[indices,:,:,:]
    y_train = y[indices, :, :]

    print X.shape, y.shape, np.min(X), np.max(X), np.min(y), np.max(y)
    print X_train.shape, y_train.shape

    np.save("/s/red/a/nobackup/cwc/palm_recognition/dataset/X_train_depth_merged.npy", X_train)
    np.save("/s/red/a/nobackup/cwc/palm_recognition/dataset/y_train_depth_merged.npy", y_train)

def map_to_rgb(l):
    if l is not None:
        l = l * 0.93
        l += 14
    return l

def save_dataset_rgb():
    labels_root = "/s/red/a/nobackup/cwc/palm_recognition/palm_labels/"
    labels = np.vstack([np.load(os.path.join(labels_root,numpy_file)) for numpy_file in os.listdir(labels_root)])


    index_array = np.zeros((240/8, 320/8, 2))

    for i in range(index_array.shape[0]):
        for j in range(index_array.shape[1]):
            index_array[i][j] = [i, j]

    X = []
    y = []
    count = 0
    for file_name, l_x, l_y, r_x, r_y in labels:

        file_name = file_name.replace("K","M")

        l_x, l_y, r_x, r_y = map_to_rgb(l_x), map_to_rgb(l_y), map_to_rgb(r_x), map_to_rgb(r_y)


        left = [l/8  if l is not None else l for l in [l_x, l_y]]
        right = [r/8 if r is not None else r for r in [r_x, r_y]]

        print count, file_name, left, right

        heat_map = get_heat_map(index_array, left, right)

        image = skimage.io.imread(file_name)

        '''image = rescale(image, 1 / 8.0)

        plt.subplot(121)
        plt.imshow(image, cmap="gray")
        plt.imshow(heat_map[:, :, 0], alpha=0.5)

        plt.subplot(122)
        plt.imshow(image, cmap="gray")
        plt.imshow(heat_map[:, :, 1], alpha=0.5)
        plt.show()'''

        X.append(image)
        y.append(heat_map)

        count += 1


    X = np.stack(X)
    y = np.stack(y)


    indices = range(X.shape[0])
    shuffle(indices)
    print indices

    X_train = X[indices,:,:,:]
    y_train = y[indices, :, :]

    print X.shape, y.shape, np.min(X), np.max(X), np.min(y), np.max(y)
    print X_train.shape, y_train.shape

    np.save("/s/red/a/nobackup/cwc/palm_recognition/dataset/X_train_rgb_merged.npy", X_train)
    np.save("/s/red/a/nobackup/cwc/palm_recognition/dataset/y_train_rgb_merged.npy", y_train)


def get_mean_image():
    sum = np.zeros((240,320, 3))
    count  = 0
    for i in range(1,35879):
        folder = ((i - 1) / 200) + 1
        root = "/s/red/a/nobackup/cwc/skeleton/ChaLearn17/frames/train/%03d/M_%05d"%(folder,i)
        print folder, i

        frame_list = os.listdir(root)
        shuffle(frame_list)
        for f in frame_list[:5]:
            sum += skimage.io.imread(os.path.join(root, f))
            count += 1

    mean = sum/count
    np.save("/s/red/a/nobackup/cwc/palm_recognition/dataset/mean_rgb.npy",mean)
    print np.max(mean), np.min(mean)
    plt.imshow(mean,cmap="gray")
    plt.show()

def merge_heat_maps():
    images = np.load("/s/red/a/nobackup/cwc/palm_recognition/dataset/X_train_rgb.npy")
    heat_maps = np.load("/s/red/a/nobackup/cwc/palm_recognition/dataset/y_train_depth.npy")
    merged_heat_maps = np.sum(heat_maps,axis=3)
    np.save("/s/red/a/nobackup/cwc/palm_recognition/dataset/y_train_depth_merged.npy", merged_heat_maps)
    print images.shape, heat_maps.shape, merged_heat_maps.shape

    '''for index, map in enumerate(heat_maps):
        index1, index2 = np.argmax(map[:,:,0]), np.argmax(map[:,:,1])
        if index1!=0 and index2!=0 and np.abs(index1-index2)<3:
            print index1, index2
            plt.imshow(images[index])
            plt.show()

            plt.subplot(121)
            plt.imshow(map[:,:,0])
            plt.subplot(122)
            plt.imshow(map[:, :, 1])
            plt.show()

            plt.imshow(map[:, :, 0]+ map[:, :, 1])
            plt.show()'''



if __name__ == "__main__":
    #save_dataset("Depth")
    save_dataset()
    #save_dataset_rgb()
    #get_mean_image()
    #merge_heat_maps()