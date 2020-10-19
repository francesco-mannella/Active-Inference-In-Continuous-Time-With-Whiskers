import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import glob


class vidManager:
    """
    Collects images and make a gif video
    """

    def __init__(self, fig, name="vid", dirname="frames", duration=300):
        """
        Args:
            fig: matplotlib.pyplot.figure, figure object where to draw images
            name: str, root name of the images the giv video will be <name>.gif
            dirname: str, the name of the folder in which images and the video will be saved
            duration: int or tuple, The display duration of each frame of the multiframe
                      gif, in milliseconds. Pass a single integer for a constant
                      duration, or a list or tuple to set the duration for each
                      frame separately.
        """
        self.name = name
        self.fig = fig
        self.dir = dirname
        self.duration = duration
        self.clear()

    def clear(self):
        """
        Clear all stuff within the folder (create it if needed)
        """
        self.t = 0
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
        files = glob.glob(self.dir + os.sep + "*.png")
        for f in files:
            if(os.path.isfile(f)):
                os.remove(f)

    def save_frame(self):
        """
        Save a single frame as an image
        """

        self.fig.canvas.draw()
        self.fig.savefig(self.dir + os.sep + self.name + "%08d.png" % self.t)
        self.t += 1

    def mk_video(self):
        """
        Make a gif file from saved frames. the gif file will be in
        <self.dir>/<self.name>.gif
        """
        # Create the frames
        frames = []
        imgs = glob.glob(self.dir + os.sep + self.name + "*.png")
        for i in sorted(imgs):
            new_frame = Image.open(i)
            frames.append(new_frame)

        # Save into a GIF file that loops forever
        frames[0].save(self.dir + os.sep + self.name + '.gif',
                       format='GIF',
                       append_images=frames[1:],
                       save_all=True,
                       duration=self.duration, loop=0)


if __name__ == "__main__":

    # USAGE

    # prepare graphics
    fig = plt.Figure()
    ax = fig.add_subplot(111)
    pnt = ax.scatter(0, 0, s=100, color="red")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    # init video maker
    vm = vidManager(fig, "fooname", "foodir")

    for p in np.linspace(0, 1, 30):

        # update fig
        pnt.set_offsets([p, p])
        vm.save_frame()

    vm.mk_video()
