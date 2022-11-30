import os
import time
import pafy
import cv2
import shutil
from configparser import ConfigParser
from yolo import YOLO
from tracker import Tracker, TrackedObject
from animal import Animal
from draw import Draw

os.environ["PAFY_BACKEND"] = "internal"

if __name__ == "__main__":
    with open("config.cfg", "r") as file:
        config = ConfigParser()
        config.read_file(file)

    video = config["Parameters"]["video"]
    stream = video[:4] == "http"

    if stream:
        name = "stream"
    else:
        name = os.path.splitext(video.split("/")[-1])[0]

    if not os.path.exists("tracks"):
        os.makedirs("tracks")

    if os.path.exists(f"tracks/tracks_{name}"):
        shutil.rmtree(f"tracks/tracks_{name}")

    os.makedirs(f"tracks/tracks_{name}")
    output_folder = config["Parameters"]["output_folder"]

    if output_folder != "None":
        if len(os.listdir(output_folder)) > 0:
            start_name = sorted([int(os.path.splitext(i)[0]) for i in os.listdir(output_folder)])[-1] + 1
        else:
            start_name = 1

        vw = cv2.VideoWriter(f"{output_folder}/{start_name}.mp4", cv2.VideoWriter_fourcc("M", "P", "4", "V"), 30,
                             (1920, 1080))

    yolo = YOLO()
    tracker = Tracker(max_disappeared=60, max_distance=200)
    tracked_objects = {}
    index = 1

    if stream:
        vc = cv2.VideoCapture(pafy.new(video).getbest(preftype="mp4").url)
    else:
        vc = cv2.VideoCapture(video)

    while True:
        returned, frame = vc.read()

        if returned:
            visualization = frame.copy()
            detections = yolo.forward(frame)
            centroids = []
            images = []

            for detection in detections:
                centroids.append(YOLO.get_centroid(detection))
                images.append(YOLO.get_crop(frame, detection))

            objects, colors = tracker.update(centroids)
            animals = Animal.animal_factory(objects, centroids, detections, images, colors)

            for animal in animals:
                TrackedObject.update(tracked_objects, animal)
                TrackedObject.visualize_track(visualization, tracked_objects, animal, 20)
                Draw.bounding_box(visualization, animal)
                Draw.animal_id(visualization, animal)

            cv2.imshow("animal-analysis", visualization)

            if output_folder != "None":
                vw.write(visualization)

            for animal in animals:
                if animal.image is not None:
                    cv2.imshow(f"Track: {animal.object_id}", animal.image)

                    if not os.path.exists(f"tracks/tracks_{name}/{animal.object_id}_" +
                                          f"{tracked_objects[animal.object_id].specie}"):
                        os.makedirs(f"tracks/tracks_{name}/{animal.object_id}_" +
                                    f"{tracked_objects[animal.object_id].specie}")

                    cv2.imwrite(f"tracks/tracks_{name}/{animal.object_id}_{tracked_objects[animal.object_id].specie}/" +
                                f"{len(tracked_objects[animal.object_id].centroids)}.jpg", animal.image)

            if stream:
                key = cv2.waitKey(1)
            else:
                key = cv2.waitKey(10)

            if key == 27:
                vc.release()
                cv2.destroyAllWindows()
                break
        else:
            if stream:
                # Stream failed. Try to re-launch the stream.
                vc.release()

                if output_folder != "None":
                    vw.release()

                cv2.destroyAllWindows()
                time.sleep(10)
                vc = cv2.VideoCapture(pafy.new(video).getbest(preftype="mp4").url)

                if output_folder != "None":
                    vw = cv2.VideoWriter(f"{output_folder}/{start_name}.mp4",
                                         cv2.VideoWriter_fourcc("M", "P", "4", "V"), 30, (1920, 1080))
            else:
                vc.release()
                cv2.destroyAllWindows()

                if output_folder != "None":
                    vw.release()

                break

        if stream:
            if index % 2000 == 0:
                if output_folder != "None":
                    vw.release()
                    start_name += 1
                    vw = cv2.VideoWriter(f"{output_folder}/{start_name}.mp4",
                                         cv2.VideoWriter_fourcc("M", "P", "4", "V"), 30, (1920, 1080))

        index += 1
