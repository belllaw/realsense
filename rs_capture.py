import pyrealsense2 as rs
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os



class RealsenseCamera:
    def __init__(self):
        # Configure depth and color streams
        print("Loading Intel Realsense Camera")

        self.pipeline = rs.pipeline()

        config = rs.config()
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 6)
        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 6)

        # config = rs.config()
        # config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 6)
        # or
        # config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        # Start streaming
        self.pipeline.start(config)

        align_to = rs.stream.color
        self.align = rs.align(align_to)


    def get_frame_stream(self):
        # Wait for a coherent pair of frames: depth and color
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            # If there is no frame, probably camera not connected, return False
            print("Error, impossible to get the frame, make sure that the Intel Realsense camera is correctly connected")
            return False, None, None

        # Apply filter to fill the Holes in the depth image
        spatial = rs.spatial_filter()
        spatial.set_option(rs.option.holes_fill, 3)
        filtered_depth = spatial.process(depth_frame)

        hole_filling = rs.hole_filling_filter()
        filled_depth = hole_filling.process(filtered_depth)


        # Create colormap to show the depth of the Objects
        colorizer = rs.colorizer()
        depth_colormap = np.asanyarray(colorizer.colorize(filled_depth).get_data())


        # Convert images to numpy arrays
        # distance = depth_frame.get_distance(int(50),int(50))
        # print("distance", distance)
        depth_image = np.asanyarray(filled_depth.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape

        # If depth and color resolutions are different, resize color image to match depth image for display
        if depth_colormap_dim != color_colormap_dim:
            resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
            images = np.hstack((resized_color_image, depth_colormap))
        else:
            images = np.hstack((color_image, depth_colormap))

        # Show images
        # cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        # cv2.imshow('RealSense', images)
        # cv2.waitKey(1)

        # cv2.destroyAllWindows()
        return True, color_image, depth_image, depth_colormap

    def release(self):
        self.pipeline.stop()
        #print(depth_image)

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        #depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.10), 2)

        # Stack both images horizontally

        #images = np.hstack((color_image, depth_colormap))

def image_capture(img_path, depth_path):
    real = RealsenseCamera()
    while True:
        ret, bgr_frame, depth_frame, depth_colormap = real.get_frame_stream()

        cv2.imshow("depth frame", depth_frame)
        cv2.imshow("Bgr frame", bgr_frame)

        key = cv2.waitKey(1)
        if key == 27:
            break
        elif key == ord('q'): # wait for 'q' key to save and exit
            cv2.imwrite(img_path, bgr_frame)
            np.save(depth_path , depth_frame)            
            break
            cv2.destroyAllWindows()
        

    # real.release()
    # cv2.destroyAllWindows()

def plot(img, depth, cx, cy):
    IMG = cv2.imread(img, cv2.IMREAD_COLOR)
    DEPTH = np.load(depth)
    cv2.imshow("depth frame", DEPTH)
    cv2.imshow("Bgr frame", IMG)

    while True:
        key = cv2.waitKey(1)
        if key == 27:
            cv2.destroyAllWindows()
            break
    print(DEPTH[cy,cx])



if __name__=="__main__":
    img_path = '/robot_ws/src/img.png'
    depth_path = '/robot_ws/src/depth.npy'
    image_capture(img_path, depth_path)
    plot(img_path, depth_path, 672, 313)

