import numpy as np
from scipy.ndimage.morphology import binary_dilation
import unittest
import SimpleITK as sitk
import imageio
NUM = 10000

def mock_image():    
    const = 100
    theta = np.linspace(-4 * np.pi, 4 * np.pi, NUM)
    z = np.linspace(0, 10, NUM)
    r = z**2 + 3
    x = r * np.sin(theta) + const
    y = r * np.cos(theta) + const
    img = np.zeros([256,256,15])
    for i,j,k in zip(x,y,z):
        img[int(i),int(j),int(k)]=1
    img=binary_dilation(img,iterations=2)
    return img,x,y,z

class MyTest(unittest.TestCase):

    def Stest_centerline_extracted(self):        
        img,x,y,z = mock_image()
        start_point = np.array([x[0],y[0],z[0]]).astype(int)
        end_point = np.array([x[-1],y[-1],z[-1]]).astype(int)
        from extract_centerline import extract_centerline
        centerline_list = extract_centerline(img,start_point,end_point)
        actual_x, actual_y, actual_z = centerline_list
        # expected length
        self.assertEqual(len(actual_x),385)
        self.assertEqual(len(actual_y),385)
        self.assertEqual(len(actual_z),385)
        # z should be descending
        self.assertTrue(all(np.logical_or(np.gradient(actual_z)==0,np.gradient(actual_z)<0)))

    def test_extract_slice(self):
        from extract_centerline import extract_centerline, smooth_3d_array
        from extract_slice import extract_slice

        # img,x,y,z = mock_image()
        # start_point = np.array([x[0],y[0],z[0]]).astype(int)
        # end_point = np.array([x[-1],y[-1],z[-1]]).astype(int)
        
        # centerline_list = extract_centerline(img,start_point,end_point)
        # actual_x, actual_y, actual_z = centerline_list
        # smooth_3d_array = actual_x, actual_y, actual_z
                
        #img_file = sys.argv[1]
        img_file = 'img.nii.gz'

        img_obj = sitk.ReadImage(img_file)
        coord = (151,19, 37)
        slice_center = img_obj.TransformContinuousIndexToPhysicalPoint(coord)
        slice_normal = (0.5,0.5,0)
        slice_spacing = (0.5,0.5,0.5)
        slice_size = (50,50,1)
        is_label = False
        slice_obj = extract_slice(img_obj,slice_center,slice_normal,slice_spacing,slice_size,is_label,out_value=0)
        arr = sitk.GetArrayFromImage(slice_obj)
        print(slice_center,slice_normal)
        print(arr.shape,arr.dtype,np.min(arr),np.max(arr))
        arr = arr.squeeze()
        arr = 255*((arr-np.min(arr))/(np.max(arr)-np.min(arr))).clip(0,1)
        arr = arr.astype(np.uint8)
        imageio.imwrite('slice.png',arr)
        
if __name__ == '__main__':
    unittest.main()