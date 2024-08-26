For the enviroment i have used conda and the dependecies are from https://github.com/Pointcept/SegmentAnything3D if you just follow the Installation you can run all the code.

SegmentAnything3d is based on https://github.com/Pointcept/SegmentAnything3D . Ihave modified it to take in color and depth images and outputs a segmented point cloud . You can test the code with the cwc database in onedrive link. And the commda is type (sam3d) t@t-Legion-T5-26AMR5:~/SegmentAnything3D$ python sam3d.py --rgb_path /.../cwc --save_path /../example --save_2dmask_path /.../2d --sam_checkpoint_path /.../sam_vit_h_4b8939.pth

segmentation_algorithms consists of open3d_segmentation_diffraction.ipynb and diffraction_edge both takes in point clouds opend3d .ply and diffraction_edge.py .pcd . Diffraction is based on this https://github.com/denabazazian/Edge_Extraction you can test the code with the .ply and .pcd files in the onedrive
TODO: 1. Adding the POV to the diffraction edge separete the point cloud edges from each other maybe use the segmented things to search for the segmented edges. 
2. Testing the SegemntAnything with a dataset which is similar to ours.
