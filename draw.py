import numpy as np
import cv2


class Draw:
    @staticmethod
    def bounding_box(image, animal, radius=20, length=15, thickness=4):
        if animal.detection is not None:
            x_start, y_start = animal.detection[0], animal.detection[1]
            x_end, y_end = animal.detection[2], animal.detection[3]
            cv2.line(image, (x_start + radius, y_start), (x_start + radius + length, y_start), animal.color, thickness)
            cv2.line(image, (x_start, y_start + radius), (x_start, y_start + radius + length), animal.color, thickness)
            cv2.ellipse(image, (x_start + radius, y_start + radius), (radius, radius), 180, 0, 90, animal.color,
                        thickness)
            cv2.line(image, (x_end - radius, y_start), (x_end - radius - length, y_start), animal.color, thickness)
            cv2.line(image, (x_end, y_start + radius), (x_end, y_start + radius + length), animal.color, thickness)
            cv2.ellipse(image, (x_end - radius, y_start + radius), (radius, radius), 270, 0, 90, animal.color,
                        thickness)
            cv2.line(image, (x_start + radius, y_end), (x_start + radius + length, y_end), animal.color, thickness)
            cv2.line(image, (x_start, y_end - radius), (x_start, y_end - radius - length), animal.color, thickness)
            cv2.ellipse(image, (x_start + radius, y_end - radius), (radius, radius), 90, 0, 90, animal.color, thickness)
            cv2.line(image, (x_end - radius, y_end), (x_end - radius - length, y_end), animal.color, thickness)
            cv2.line(image, (x_end, y_end - radius), (x_end, y_end - radius - length), animal.color, thickness)
            cv2.ellipse(image, (x_end - radius, y_end - radius), (radius, radius), 0, 0, 90, animal.color, thickness)

    @staticmethod
    def animal_id(image, animal):
        # if animal.detection is not None:
        label = f"#{animal.object_id}: {animal.specie}"
        border_length = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 25)
        label_length = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.putText(image, label,
                    (np.max([0, animal.centroid[0] - border_length[0][0] // 2 + 10]),
                     np.max([0, animal.centroid[1] - 30])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, tuple([i + 30 for i in animal.color]), 25, cv2.LINE_AA)
        cv2.putText(image, label,
                    (np.max([0, animal.centroid[0] - label_length[0][0] // 2]),
                     np.max([0, animal.centroid[1] - 30])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, tuple([i - 30 for i in animal.color]), 2, cv2.LINE_AA)
