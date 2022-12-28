import plotly.graph_objects as go
from plotly.subplots import make_subplots

import matplotlib.pyplot as plt
plt.style.use('ggplot')

import numpy as np

class VisPlot:
    def __init__(self, dims, num_subplots=1):
        """
            Create a figure object in order to sequentially add plots.

            :param dims: the number of dimensions (mpl for 2, plotly for 3).
            :param num_subplots: number of desired subplots.
        """

        if dims == 2:
            print("warning: 2d plots don't really work yet.")
            fig = plt.figure()
            for i in range(num_subplots):
                fig.add_subplot(1, num_subplots, i + 1)
        elif dims == 3:
            camera = dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=0.5, y=1.25, z=0)
            )

            fig = make_subplots(rows=1, cols=num_subplots, specs = [[{'is_3d':True}]] * num_subplots)
            
            fig.update_layout(scene_camera=camera)
            fig.update_scenes(xaxis_visible=False, yaxis_visible=False,zaxis_visible=False)

        self.fig = fig
        self.dims = dims
        self.num_subplots = num_subplots

    def add(self, points, size=None, color=None, labels=None, outline=False):
        """
            Plots some points onto the figure (it's successively added).

            :param points: Nxdims points to be plotted.
            :param size: arr of length N, or scalar, that specifies size of points, default all 1.
            :param color: arr of size num_subplots x N x {3,4} that specify colors, default all black.
                Different subplots can have different colorings, but will have the same points, sizes, and labels.
            :param outline: (3D only) True if you want to see an outline around the points.
            :param labels: dictionary to use as labels, default no additional labels.
        """

        if len(np.shape(color)) == 2:
            lcolor = color[np.newaxis, :, :]
        else:
            lcolor = color

        assert(np.shape(lcolor)[0] == self.num_subplots) # check number of colors is eq to num subplots
        assert(np.shape(points)[1] == self.dims) # check that the number of dimensions of embeddings is correct

        if size is None:
            size = 1
        if color is None:
            color = np.array([0,0,0])
        if labels is None:
            labels = {}

        outline_width = 1 if outline else 0

        if self.dims == 2:
            for plot in range(self.num_subplots):
                ax = self.fig.add_subplot(1, np.shape(lcolor)[0], 1 + plot)
                ax.scatter(points[:, 0], points[:, 1], s=size, c=lcolor[plot])
        elif self.dims == 3:
            for plot in range(self.num_subplots):
                self.fig.add_trace(go.Scatter3d(
                    x=points[:, 0],
                    y=points[:, 1],
                    z=points[:, 2],
                    mode='markers',
                    marker=dict(
                        size=size,
                        color=lcolor[plot],
                        line=dict(width=outline_width, color='black'),
                        opacity=1
                    ),
                    text=[{k: v[i] for k, v in labels.items()} for i in range(np.shape(points)[0])]
                ), 1, plot + 1)

    def show(self):
        """
            Displays figure that's been successively plotted.
        """

        if self.dims == 2:
            plt.show()
        elif self.dims == 3:
            self.fig.write_html('plot.html', auto_open=True)
