'''
main.py
'''
import cv2
from matplotlib import pyplot as plt
from imutils.perspective import four_point_transform
from imutils import contours
import imutils
import numpy as np

def preprocess_image():
    '''
    preprocess_image fct - generate initial images
    '''
    #image = cv2.imread('dataset/exemple_corecte/image_23.jpg')
    #image = cv2.imread('dataset/exemple_corecte/image_7.jpg')
    image = cv2.imread('dataset/exemple_corecte/rotation_24.jpg')
    #image = cv2.imread('dataset/exemple_corecte/perspective_24.jpg')
    #image = cv2.imread('dataset/rotation_207.jpg')
    image_width = int(image.shape[1])
    image_height = int(image.shape[0])
    resize_ratio = min([int(image_width / 720), int(image_height / 1280)])
    if resize_ratio == 0:
        resize_ratio = 1
    image = cv2.resize(image, (int(image.shape[1] / resize_ratio), \
                       int(image.shape[0] / resize_ratio)), interpolation=cv2.INTER_CUBIC)
    original_image = image.copy()
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    edged_image = cv2.Canny(blurred_image, 30, 100)

    return image, original_image, gray_image, edged_image

def detect_answer_boxes(answer_boxes_image, original_image_contours):
    count_squares = 0
    answer_boxes = []
    if len(original_image_contours) > 0:
        original_image_contours = sorted(original_image_contours, \
                                        key=cv2.contourArea, reverse=True)
        for contour in original_image_contours:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
            approx = approx.reshape((-1, 1, 2))
            if len(approx) == 4:
                cv2.polylines(answer_boxes_image, [approx], True, (0, 255, 255), 2)
                count_squares = count_squares + 1
                answer_boxes.append(approx)
                if count_squares >= 2:
                    break

    if len(answer_boxes) != 2:
        raise Exception

    # sort: math answers box before info/physics answers box
    if answer_boxes[1].min(axis=0)[0][0] < answer_boxes[0].min(axis=0)[0][0]:
        answer_boxes = answer_boxes[::-1]

    return answer_boxes_image, answer_boxes

def crop_image(image, y_start, x_start):
    '''
    crop an image starting from (x_start, y_start) to right-bottom corner
    '''
    cropped_image = image.copy()
    cropped_image = cropped_image[y_start:, x_start:]
    return cropped_image

def scale_contour(cnt, scale):
    M = cv2.moments(cnt)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])

    cnt_norm = cnt - [cx, cy]
    cnt_scaled = cnt_norm * scale
    cnt_scaled = cnt_scaled + [cx, cy]
    cnt_scaled = cnt_scaled.astype(np.int32)

    return cnt_scaled

def main():
    '''
    main function
    '''
    # basic image pre-processing: gray scale/canny
    answer_boxes_image, original_image, gray_image, edged_image = preprocess_image()

    # all contours inside original image
    original_image_contours = cv2.findContours(edged_image.copy(), cv2.RETR_TREE, \
                                              cv2.CHAIN_APPROX_SIMPLE)
    original_image_contours = imutils.grab_contours(original_image_contours)
    contours_original_image = original_image.copy()
    cv2.drawContours(contours_original_image, original_image_contours, -1, \
                     (0, 255, 255), thickness=2)

    # original image with answer boxes marked and boxed coordinates
    answer_boxes_image, answer_boxes = detect_answer_boxes(answer_boxes_image, \
                                                           original_image_contours)

    # one image for every answer box (math/info)
    answer_box_images = []
    answer_box_gray_images = []
    # homomorphic transformation (for rotation/perspective part)
    for answer_box in answer_boxes:
        answer_box_images.append(four_point_transform(original_image.copy(), \
                                                      answer_box.reshape(4, 2)))
        answer_box_gray_images.append(four_point_transform(gray_image.copy(), \
                                                           answer_box.reshape(4, 2)))

    # answer boxes heights - one answer box == 8x17 option squares
    answer_boxes_heights = [round(len(answer_box_images[0]) / 17), \
                        round(len(answer_box_images[1]) / 17)]
    answer_boxes_widths = [round(len(answer_box_images[0][0]) / 8), \
                        round(len(answer_box_images[1][0]) / 8)]

    # get Info/Fiz portion of image
    optional_test_answer_box = answer_boxes[1]
    optional_box_height = len(answer_box_images[1])
    scale = 2 * 4 * answer_boxes_heights[1] / optional_box_height + 1
    width_scale_adaos = int(len(answer_box_images[1][0]) * (scale - 1) / 2)
    optional_test_answer_box = scale_contour(optional_test_answer_box, scale)
    test_chosen_option_image = four_point_transform(original_image.copy(), \
                                                    optional_test_answer_box.reshape(4, 2))
    # get upper part where Info/Fizica text is located
    test_chosen_option_image = test_chosen_option_image[: 4 * answer_boxes_heights[1], :]
    test_chosen_option_image = test_chosen_option_image[:, width_scale_adaos :-width_scale_adaos]
    test_chosen_option_image = test_chosen_option_image[:, -2 * answer_boxes_widths[1]:]
    original_options_image = test_chosen_option_image.copy()
    test_chosen_option_image = cv2.cvtColor(test_chosen_option_image, cv2.COLOR_BGR2GRAY)
    test_chosen_option_image = cv2.threshold(test_chosen_option_image, 0, 255, \
                                             cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # horizontal lines removal
    lines = cv2.HoughLines(test_chosen_option_image, 1, np.pi/180, \
                           int(1.1 * answer_boxes_heights[1]))
    for rho, theta in lines.reshape(-1, 2):
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        angle = np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi
        if abs(angle) < 2 or abs(angle) > 88 and abs(angle) < 92:
            cv2.line(test_chosen_option_image, (x1, y1), (x2, y2), (0, 0, 0), 2)

    test_chosen_contours = cv2.findContours(test_chosen_option_image, cv2.RETR_TREE, \
        cv2.CHAIN_APPROX_SIMPLE)
    test_chosen_contours = imutils.grab_contours(test_chosen_contours)

    # option detection - low precision
    selected_option = 3
    test_option_contours = np.array([])
    for contour in test_chosen_contours:
        (_, _, width, height) = cv2.boundingRect(contour)
        aspect_ratio = height / float(width)
        if height >= 0.2*answer_boxes_heights[1]:
            test_option_contours = contour
            if aspect_ratio > 0.9 and aspect_ratio < 1.1:
                selected_option = 2
            if aspect_ratio < 0.5:
                selected_option = 1
            if aspect_ratio > 0.5 and aspect_ratio < 0.9:
                selected_option = 4
            break

    chosen_option_letter = 'I'
    if test_option_contours.size:
        original_options_image = cv2.drawContours(original_options_image, \
                test_option_contours, -1, (255, 255, 0), 2)
        option_selected_bottom_coords = \
                tuple(test_option_contours[test_option_contours[:, :, 1].argmax()][0])
        if option_selected_bottom_coords[1] > len(original_options_image[0]) / 2:
            chosen_option_letter = 'F'

    #chosen_option_letter = 'I'
    #selected_option = 1

    # crop useless part of the answer box
    cropped_answer_box_images = []
    cropped_answer_box_gray_images = []
    for index, image in enumerate(answer_box_images):
        cropped_answer_box_images.append(crop_image(image, 2 * answer_boxes_heights[index], \
                                                    int(3.6 * answer_boxes_widths[index])))
    for index, image in enumerate(answer_box_gray_images):
        cropped_answer_box_gray_images.append(crop_image(image, 2 * answer_boxes_heights[index], \
                                                         int(3.6 * answer_boxes_widths[index])))

    # binary treshold on cropped_answer_box_gray_images
    answer_box_thresh_gray_images = []
    for image in cropped_answer_box_gray_images:
        answer_box_thresh_gray_images.append(cv2.threshold(image.copy(), 0, 255, \
            cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1])

    # remove box lines in order to expose X marks
    for index, image in enumerate(answer_box_thresh_gray_images):
        lines = cv2.HoughLines(image, 1, np.pi/180, int(answer_boxes_widths[index] * 3))
        for rho, theta in lines.reshape(-1, 2):
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            angle = np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi
            if abs(angle) < 2 or abs(angle) > 88 and abs(angle) < 92:
                cv2.line(answer_box_thresh_gray_images[index], (x1, y1), (x2, y2), (0, 0, 0), 3)

    # detect contours on answer_box_thresh_gray_images
    thresh_box_images_contours = []
    for thresh_image in answer_box_thresh_gray_images:
        contour = cv2.findContours(thresh_image, cv2.RETR_TREE, \
            cv2.CHAIN_APPROX_SIMPLE)
        contour = imutils.grab_contours(contour)
        thresh_box_images_contours.append(contour)

    # detect X marks
    marked_answers_contours = [[], []]
    for index, thresh_contour in enumerate(thresh_box_images_contours):
        for contour in thresh_contour:
            (_, _, width, height) = cv2.boundingRect(contour)
            aspect_ratio = height / float(width)
            if width <= answer_boxes_widths[index] and height <= answer_boxes_heights[index] and \
            width >= 0.2*answer_boxes_widths[index] and height >= 0.2*answer_boxes_heights[index]:
                marked_answers_contours[index].append(contour)

    # highlight x marks inside answer boxes
    answer_box_x_marks_images = []
    for index, contour in enumerate(marked_answers_contours):
        answer_box_x_marks_images.append(cv2.drawContours(cropped_answer_box_images[index].copy(), \
            contour, -1, (255, 255, 0), 2))

    # parse contours and determine answers
    test_answers = [{}, {}]
    for box_index, box_answers_contours in enumerate(marked_answers_contours):
        box_answers_contours = contours.sort_contours(box_answers_contours, 'top-to-bottom')[0]
        for contour in box_answers_contours:
            left_up_coords = contour.min(axis=0)
            right_down_coords = contour.max(axis=0)
            center_coords = (left_up_coords + right_down_coords) / 2
            question_number = center_coords / answer_boxes_heights[box_index]
            question_number = question_number.reshape(2,).astype(int)[1]
            question_option = center_coords / answer_boxes_widths[box_index]
            question_option = question_option.reshape(2,).astype(int)[0]
            question_option = chr(ord('A') + question_option)
            if question_number in test_answers[box_index].keys():
                test_answers[box_index][question_number] = 'Z'
            else:
                test_answers[box_index][question_number] = question_option

    # write answers inside file
    output_file = open('output_file.txt', 'w')
    output_file.write('{} {}\n'.format(chosen_option_letter, selected_option))
    parsed_dict = {}
    for box_index, box_test_answers in enumerate(test_answers):
        question_number = 1
        if box_index == 1:
            question_number = 16
        for key in box_test_answers.keys():
            letter_option = box_test_answers[key]
            if letter_option == 'Z':
                continue
            output_file.write('{} {}\n'.format(question_number+key, letter_option))
            parsed_dict[question_number+key] = letter_option

    # verify answers
    barem_dict = {}
    with open('dataset/barem/{}{}.txt'.format(chosen_option_letter, selected_option),'r') as f:
        for line in f:
            row = line.split()
            if len(row) == 2:
                if row[0].isnumeric():
                    barem_dict[int(row[0])] = row[1]

    shared_items = {k: parsed_dict[k] \
                    for k in parsed_dict if k in barem_dict and parsed_dict[k] == barem_dict[k]}
    output_file.write('R {}\n'.format(len(shared_items)))
    output_file.close()

    # plot entire flow
    cv2.cvtColor(contours_original_image, cv2.COLOR_BGR2RGB)
    answer_box_x_marks_images[0] = cv2.cvtColor(answer_box_x_marks_images[0], cv2.COLOR_BGR2RGB)
    answer_box_x_marks_images[1] = cv2.cvtColor(answer_box_x_marks_images[1], cv2.COLOR_BGR2RGB)
    answer_box_thresh_gray_images[0] = \
        cv2.cvtColor(answer_box_thresh_gray_images[0], cv2.COLOR_GRAY2RGB)
    answer_box_thresh_gray_images[1] = \
        cv2.cvtColor(answer_box_thresh_gray_images[1], cv2.COLOR_GRAY2RGB)
    original_options_image = cv2.cvtColor(original_options_image, cv2.COLOR_BGR2RGB)
    _ = plt.subplot(261), plt.imshow(original_image)
    _ = plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    _ = plt.subplot(262), plt.imshow(edged_image, cmap='gray')
    _ = plt.title('Edged Image'), plt.xticks([]), plt.yticks([])
    _ = plt.subplot(263), plt.imshow(contours_original_image)
    _ = plt.title('Contours'), plt.xticks([]), plt.yticks([])
    _ = plt.subplot(264), plt.imshow(answer_boxes_image)
    _ = plt.title('Two squares'), plt.xticks([]), plt.yticks([])
    _ = plt.subplot(265), plt.imshow(cropped_answer_box_images[0])
    _ = plt.title('First box'), plt.xticks([]), plt.yticks([])
    _ = plt.subplot(266), plt.imshow(cropped_answer_box_images[1])
    _ = plt.title('Second box'), plt.xticks([]), plt.yticks([])
    _ = plt.subplot(267), plt.imshow(answer_box_thresh_gray_images[0])
    _ = plt.title('Thresh 1st box'), plt.xticks([]), plt.yticks([])
    _ = plt.subplot(268), plt.imshow(answer_box_thresh_gray_images[1])
    _ = plt.title('Thresh 2nd box'), plt.xticks([]), plt.yticks([])
    _ = plt.subplot(269), plt.imshow(answer_box_x_marks_images[0])
    _ = plt.title('X Marks 1st box'), plt.xticks([]), plt.yticks([])
    _ = plt.subplot(2, 6, 10), plt.imshow(answer_box_x_marks_images[1])
    _ = plt.title('X Marks 2nd box'), plt.xticks([]), plt.yticks([])
    _ = plt.subplot(2, 6, 11), plt.imshow(test_chosen_option_image, cmap='gray')
    _ = plt.title('Test option thresh'), plt.xticks([]), plt.yticks([])
    _ = plt.subplot(2, 6, 12), plt.imshow(original_options_image)
    _ = plt.title('Test option contour'), plt.xticks([]), plt.yticks([])
    plt.show()

if __name__ == '__main__':
    main()
