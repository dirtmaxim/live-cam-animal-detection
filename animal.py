import numpy as np


class Animal:
    def __init__(self, object_id=None, centroid=None, detection=None, image=None, color=None):

        self.object_id = object_id
        self.centroid = centroid
        self.color = color
        self.classes = {
            0: "North. Card.",
            1: "North. Card.",
            2: "Blue Jay",
            3: "American Crow",
            4: "Red-wing. Black.",
            5: "Red-wing. Black.",
            6: "Common Grackle",
            7: "Common Grackle",
            8: "House Sparrow",
            9: "Mourning Dove",
            10: "Mallard Duck",
            11: "Mallard Duck",
            12: "Virginia Opossum",
            13: "East. Cottontail",
            14: "East. Grey Squirrel",
            15: "American Red Squirrel",
            16: "East. Chipmunk",
            17: "White-tail. Deer",
            18: "Red Fox",
            19: "Groundhog",
            20: "Raccoon"
        }

        if detection is not None:
            self.detection = (int(detection[0]), int(detection[1]), int(detection[2]), int(detection[3]))
            self.specie = self.classes[detection[5]]
            self.image = image
        else:
            self.detection = None
            self.specie = "?"
            self.image = None

    @staticmethod
    def animal_factory(objects, centroids, detections, images, colors):
        animals = []

        for object_id, centroid in objects.items():
            assigned = False

            for i, c in enumerate(centroids):
                if np.array_equal(centroid, c):
                    animals.append(Animal(object_id, centroid, detections[i], images[i], colors[object_id]))
                    assigned = True

            if not assigned:
                # Object disappeared for some frames.
                animals.append(Animal(object_id, centroid, None, None, colors[object_id]))

        return animals
