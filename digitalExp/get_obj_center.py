import cv2

def get_obj_center(img_path):
    """
    返回图片中目标的中心点
    :param img_path:
    :return:
    """
    #读取图片
    bgr_img = cv2.imread(img_path)
    #二值化
    gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

    # blurred = cv2.GaussianBlur(gray_img, (9, 9),0)
    gradX = cv2.Sobel(gray_img, ddepth=cv2.CV_32F, dx=1, dy=0)
    gradY = cv2.Sobel(gray_img, ddepth=cv2.CV_32F, dx=0, dy=1)
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)
    blurred = cv2.GaussianBlur(gradient, (9, 9),0)


    th, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_OTSU)
    #获取轮廓的点集
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #取最大的边缘轮廓点集
    contours = max(contours, key=cv2.contourArea)

    #求取轮廓的矩
    M = cv2.moments(contours)

    #画出轮廓
    cv2.drawContours(bgr_img, contours, -1, (0, 0, 255), 3)
    bounding_boxes = [cv2.boundingRect(cnt) for cnt in contours]

    #在图片上画出矩形边框
    # for bbox in bounding_boxes:
    #     [x, y, w, h] = bbox
    #     cv2.rectangle(bgr_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #通过矩来计算轮廓的中心坐标
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    # print(f"({cx},{cy})")
    # cv2.circle(bgr_img,(cx,cy), 4, (255,255,0), thickness=8)
    # cv2.imshow("name", bgr_img)
    # cv2.waitKey(0)
    print("have got obj center:",cx, cy)
    return cx, cy

# cx, cy = get_obj_center('ILSVRC2012_val_00004021.JPEG')
# print(cx,cy)