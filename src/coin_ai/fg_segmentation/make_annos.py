import argparse
import os
import numpy as np
import imageio
import matplotlib.pyplot as plt


class Annotator:
    def __init__(self, image_path: str, dst_path: str):
        self.image_path = image_path
        self.dst_path = dst_path
        self.points = []

    def collect(self):
        image = imageio.imread(self.image_path)
        fig, ax = plt.subplots()
        ax.imshow(image)

        def onclick(event):
            if event.dblclick:
                self.points.append([event.xdata, event.ydata])
                ax.plot(event.xdata, event.ydata, "ro")
                fig.canvas.draw()

        fig.canvas.mpl_connect("button_press_event", onclick)

        plt.show()

    def save(self):
        np.save(self.dst_path, np.array(self.points))
        print(f"Saved {len(self.points)} points to {self.dst_path}")

    def run(self):
        self.collect()
        self.save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--dst", type=str, required=True)
    args = parser.parse_args()

    src_paths = [f for f in os.listdir(args.root) if f.lower().endswith("jpg")]

    for src_path in src_paths:
        dst_path = src_path.removeprefix(args.root).removesuffix(".jpg")
        dst_path = os.path.join(args.dst, dst_path + ".npy")

        annotator = Annotator(os.path.join(args.root, src_path), dst_path)
        print(annotator.run())
