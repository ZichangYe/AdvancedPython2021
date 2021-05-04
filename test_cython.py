
from video_transformer_test_cython import *
import os
# test_2 = video_transformer_base(path=path, 
#                           save_path=os.path.join(path, "data", "frames"),
#                           file_name='hamilton_clip.mp4', 
#                           display=False,
#                           device='cpu')
# test_2.main_transformation('mtcnn',test_2.mean_blur)
## worked! 
# test_2 = video_transformer_numba(path=path, 
#                           save_path=os.path.join(path, "data", "frames"),
#                           file_name='hamilton_clip.mp4', 
#                           display=False,
#                           device='cpu')
# test_2.main_transformation('mtcnn',test_2.mean_blur)
## worked! 
test_3 = video_transformer_parallel(path=path, 
                          save_path= os.path.join(path, "data", "frames"),
                          file_name='dance.mp4', 
                          display=False,
                          device='cpu')


test_3.filter_on_video(test_3.mean_blur)
test_3.write_to_video()