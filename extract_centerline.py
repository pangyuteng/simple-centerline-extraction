import numpy as np
import skfmm
from scipy.ndimage import distance_transform_edt
from scipy.interpolate import UnivariateSpline

def get_boundary_seeded_field(img):
    bs_field = distance_transform_edt(img>0)
    return bs_field

def get_point_seeded_field(img,seed):
    sx,sy,sz = seed
    mask = ~img.astype(bool)
    img = img.astype(float)
    m = np.ones_like(img)
    m[sx,sy,sz] = 0
    m = np.ma.masked_array(m, mask)
    ss_field = skfmm.distance(m)
    return ss_field

def extract_centerline(img,start_point,end_point,search_radius=(2,2,2),return_multi_index=True):
    # img: binary mask
    # start_point: used to generate ss field
    # end_point: start of travesel of centerline
    
    sr = search_radius
    bs_field = get_boundary_seeded_field(img)
    ss_field = get_point_seeded_field(img,start_point)
    shape = ss_field.shape    
    
    start_point_ind = np.ravel_multi_index(start_point,shape)
    current_point_ind = np.ravel_multi_index(end_point,shape)
    
    centerline_list=[]
    centerline_list.append(current_point_ind)
    while np.take(ss_field,centerline_list[-1]) > np.take(ss_field,start_point_ind):
        # get search region (ugly...)
        cp_ind = centerline_list[-1]
        cp = np.unravel_index(cp_ind,shape)        
        Xs = cp[0]-sr[0]
        Xe = cp[0]+sr[0]+1 
        Ys = cp[1]-sr[1]
        Ye = cp[1]+sr[1]+1
        Zs = cp[2]-sr[2]
        Ze = cp[2]+sr[2]+1

        xs = np.arange(Xs,Xe,1)
        ys = np.arange(Ys,Ye,1)
        zs = np.arange(Zs,Ze,1)
        grid = np.meshgrid(xs,ys,zs)
        
        multi_index = np.array([grid[0].ravel(),grid[1].ravel(),grid[2].ravel()])
        region_ind = np.ravel_multi_index(multi_index,shape,mode='clip')
        bs_region = np.take(bs_field,region_ind,mode='clip')
        ss_region = np.take(ss_field,region_ind,mode='clip')
        
        # within search region, find minimum ss location (closer to start point)
        min_ss = np.min(ss_region)
        bs_region_proc = np.array(bs_region)
        
        # within region where equal to min ss value, find max bs value (most centered)
        bs_region_proc[ss_region!=min_ss]=0.0
        regional_new_point_ind = np.argmax(bs_region_proc)
        
        # assign most centered and closer to start point location as next point in centerline.
        new_point_ind = region_ind[regional_new_point_ind]
        new_point = np.unravel_index(new_point_ind,shape)
        
        if new_point_ind in centerline_list:
            break
            
        centerline_list.append(new_point_ind)
        if len(centerline_list) > 3:
            # as we traverse in the `ss map`, `ss` value in last point (forefront) should be smaller than prior points, and end_point should have the highest `ss` value from all centerline points. we `break` the while loop failing this criteria.
            if np.take(ss_field,centerline_list[-2]) < np.take(ss_field,centerline_list[-1]):
                break
                
    if return_multi_index:
        return np.unravel_index(centerline_list,shape)
    else:
        return centerline_list


def smooth_3d_array(x,y,z,num=None,**kwargs):
    if num is None:
        num = len(x)
    w = np.arange(0,len(x),1)
    sx = UnivariateSpline(w,x,**kwargs)
    sy = UnivariateSpline(w,y,**kwargs)
    sz = UnivariateSpline(w,z,**kwargs)
    wnew = np.linspace(0,len(x),num)
    return sx(wnew),sy(wnew),sz(wnew)

"""

TODO:
replace UnivariateSpline with csaps
de Boor, A Practical Guide to Splines, Springer-Verlag, 1978 
https://csaps.readthedocs.io/en/latest 
https://www.mathworks.com/help/curvefit/csaps.html

"""
