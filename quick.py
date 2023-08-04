import pyrealsense2 as rs
import numpy as np
import cv2 as cv

pipeline = rs.pipeline()    # realsense pipeline open
config = rs.config()    # config class 생성
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
# stream 종류, size, format 설정등록
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

#pipeline.start()    # Start streaming
profile = pipeline.start(config)   # pipeline start

depth_sensor = profile.get_device().first_depth_sensor()    # depth sensor에 대한 것들을 얻자
depth_scale = depth_sensor.get_depth_scale()    # 깊이 센서의 깊이 스케일 얻음
print("Depth Scale is: ", depth_scale)

clipping_distance_in_meters = 1    # 1 meter, 클리핑할 영역을 1m로 설정
clipping_distance = clipping_distance_in_meters / depth_scale   #스케일에 따른 클리핑 거리



try:
    while True:
        # Camera Start (external)-------------------------------------------------------------------------------

        frames = pipeline.wait_for_frames()
        # color와 depth의 프레임셋을 기다림

        # frames.get_depth_frame() 은 640x480 depth 이미지
        depth_frame = frames.get_depth_frame()
        # depth 프레임은 640x480 의 depth 이미지
        color_frame = frames.get_color_frame()

        #프레임이 없으면, 건너 뜀
        if not depth_frame or not color_frame:
                continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data()) # depth 이미지를 배열로
        color_image = np.asanyarray(color_frame.get_data()) # color 이미지를 배열로

        #백그라운드 제거
        grey_color = 153
        depth_image_3d = np.dstack((depth_image, depth_image, depth_image))  #depth image는 1채널, 컬러 이미지는 3채널
        bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)
        # 클리핑 거리를 깊이 _이미지가 넘어서거나, 0보다 적으면, 회색으로 아니면 컬러 이미지로 반환

        #이미지 렌더링
        depth_colormap = cv.applyColorMap(cv.convertScaleAbs(depth_image, alpha=0.03), cv.COLORMAP_JET)
        # applyColorMap(src, 필터) 필터를 적용함 , COLORMAP_JET=  연속적인 색상, blue -> red
        # convertScaleAbs: 인자적용 후 절대값, 8비트 반환

        # 두 이미지를 수평으로 연결
        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape
        # If depth and color resolutions are different, resize color image to match depth image for display
        if depth_colormap_dim != color_colormap_dim:
            resized_color_image = cv.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
            images = np.hstack((resized_color_image, depth_colormap))
        else:
            images = np.hstack((color_image, depth_colormap))

        # Show images
        cv.namedWindow('RealSense', cv.WINDOW_AUTOSIZE)   #이미지 윈도우 정의
        cv.imshow('RealSense', images) #이미지를 넣어 윈도우에 보임


        '''
        camera = cv.VideoCapture(0)
        frame_width2 = int(camera.get(3))
        frame_height2 = int(camera.get(4))
        size2 = (frame_width2, frame_height2)
        result_camera = cv.VideoWriter('NameAsDateTime_camera.avi', cv.VideoWriter_fourcc(*'MJPG'), 10, size2)
        while(camera.isOpened()):
            ret_cam, frame_cam = camera.read()
            if ret_cam == True:
                result_camera.write(color_image)
                cv.imshow("Camera", color_image)
                key_cam = cv.waitKey(20) & 0xFF
        '''

        # ESC 키를 눌렀을 경우, 프로그램을 종료
        key = cv.waitKey(1) #키 입력
        if key & 0xFF == ord('q') or key == 27: #나가기
            cv.destroyAllWindows()  #윈도우 제거
            break
finally:
     pipeline.stop()    # Stop streaming 리얼센스 데이터 스트리밍 중지



