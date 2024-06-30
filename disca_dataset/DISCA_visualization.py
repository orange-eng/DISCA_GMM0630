import pickle, mrcfile
import numpy as np
import scipy.ndimage as SN

def cut_from_whole_map__se(whole_map, se):
    if se is None:
        return None
    return whole_map[se[0,0]:se[0,1], se[1,0]:se[1,1], se[2,0]:se[2,1]]


def cut_from_whole_map(whole_map, c, siz):
    se = subvolume_center_start_end(c, map_siz=whole_map.shape, subvol_siz=siz)
    return cut_from_whole_map__se(whole_map, se)


def subvolume_center_start_end(c, map_siz, subvol_siz):
    map_siz = np.array(map_siz)
    subvol_siz = np.array(subvol_siz)

    siz_h = np.ceil(subvol_siz / 2.0)

    start = c - siz_h
    start = start.astype(int)
    end = start + subvol_siz
    end = end.astype(int)

    if any(start < 0):
        return None
    if any(end >= map_siz):
        return None

    se = np.zeros( (3,2), dtype=np.int16)
    se[:,0] = start
    se[:,1] = end

    return se


def pickle_load(path): 
    with open(path, 'rb') as f:     o = pickle.load(f, encoding='latin1') 

    return o 


def smooth(v, sigma):
    assert  sigma > 0
    return SN.gaussian_filter(input=v, sigma=sigma)



def read_mrc_numpy_vol(path):
    with mrcfile.open(path) as mrc:
        v = mrc.data
        v = v.astype(np.float32).transpose([2,1,0])
    return v



def save_mrc_numpy_vol(v, path):
    with mrcfile.new(path, overwrite=True) as mrc:
        mrc.set_data(np.float32(v).transpose(2,1,0))



def cub_img(v, view_dir=2):
    if view_dir == 0:
        vt = np.transpose(v, [1,2,0])
    elif view_dir == 1:
        vt = np.transpose(v, [2,0,1])
    elif view_dir == 2:
        vt = v
    
    row_num = vt.shape[0] + 1
    col_num = vt.shape[1] + 1
    slide_num = vt.shape[2]
    disp_len = int( np.ceil(np.sqrt(slide_num)) )
    
    slide_count = 0
    im = np.zeros( (row_num*disp_len, col_num*disp_len) ) + float('nan')
    for i in range(disp_len):
        for j in range(disp_len):
            im[(i*row_num) : ((i+1)*row_num-1),  (j*col_num) : ((j+1)*col_num-1)] = vt[:,:, slide_count]
            slide_count += 1
            
            if (slide_count >= slide_num):
                break
            
        
        if (slide_count >= slide_num):
            break
   
    
    im_v = im[np.isfinite(im)]

    if im_v.max() > im_v.min(): 
        im = (im - im_v.min()) / (im_v.max() - im_v.min())

    return {'im':im, 'vt':vt}



def save_png(m, name, normalize=True, verbose=False):

    m = np.array(m, dtype=np.float)

    mv = m[np.isfinite(m)]
    if normalize:
        # normalize intensity to 0 to 1
        if mv.max() - mv.min() > 0:
            m = (m - mv.min()) / (mv.max() - mv.min())
        else:
            m = np.zeros(m.shape)
    else:
        assert mv.min() >= 0
        assert mv.max() <= 1

    m = np.ceil(m * 65534)
    m = np.array(m, dtype=np.uint16)

    import png          # in pypng package
    png.from_array(m, 'L').save(name)



if __name__ == '__main__':

    labels = pickle_load('labels.pickle')

    infos = pickle_load('info.pickle')

    v = read_mrc_numpy_vol('emd_4603.map')

    v = (v - np.mean(v))/np.std(v)

    vs = []

    s = 32//2

    for i in range(np.max(labels) + 1):
        print(i)
        locs = np.array(infos)[labels == i]

        v_i = np.zeros_like(v)

        for j in locs:
            if j[0] == 'emd_4603_deconv_corrected.mrc': #emd_4603_deconv_corrected.mrc / emd_4603.map
                v_i[j[2][0] - s: j[2][0] + s, j[2][1] - s: j[2][1] + s, j[2][2] - s: j[2][2] + s] = \
                v[j[2][0] - s: j[2][0] + s, j[2][1] - s: j[2][1] + s, j[2][2] - s: j[2][2] + s]


        save_png(cub_img(v_i[:,:,::10])['im'], str(i) + '.png')        
