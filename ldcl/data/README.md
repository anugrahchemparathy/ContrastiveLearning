# How dataset generation works

Generally, you can retrieve complete ``ConservationDataset`` objects from ``ldcl.data.physics``. Use the ``get_dataset`` function. Something like ``dataset, folder = physics.get_dataset("path_to_config.json", "saved_dataset_path/")``. Note that the second has to be adjusted if your script is changing locations within the project, so that's a bit annoying.

## Config file

The config file is ideal as it lets us save many different experiments we are running in a place where we will actually remember all the settings we used (i.e. as files). Furthermore, it'll get tracked by git, and so we can always recover old versions as well as easily being able to share experiment settings with each other.

## Top-level settings

* First, you should specify the **dynamics** you are using. This can be either ``pendulum`` or ``orbits`` at this moment. (We should add support for natural images/CIFAR sometime soon.)
* Then, **modality**. ``numerical`` is just inputting in numbers (position and velocity). ``image`` is some low-resolution artificial images I can generate.
* **Caching** is controlled by ``use_cached_data``, which is either ``True`` or ``False``. Let's try to keep this to ``True`` to save disk space, but for absolutely correct experiments no caching may be better.
* **verbose** doesn't really work, but it would control the level output.
* **orbit_settings** specifies the settings for the orbit modality.
* **pendulum_settings** and **pendulum_imagen_settings** are as you expect.

## Orbit settings

I'll document pendulum sometime too if we go back to using that frequently.

* **mu** is the gravitational parameter. Let's leave it at ``1.0`` now.
* **num_trajs** is the number of sampled trajectories. This is the total number of orbits.
* **num_ts** is the number of points sampled from each trajectory. In total, the dataset has ``num_ts * num_trajs`` data points.
* **traj_distr** is a 3D distribution that specifies the distribution of conserved quantities. The three dimensions are H, L, and phi0 in that order. In the original dataset, they lie uniformly in the intervals [-0.5,-0.25], [0,1], and [0, 2pi] respectively.
* **t_distr** is a 1D distribution that describes the distribution of times. Default this is uniform on [0,100].
* **noise** specifies the amount of noise added to each of the final output values in terms of the scale of the zero-centered Gaussian of noise.
* **pq_resample_condition** is not currently used. (It would be used to resample points according to some condition, but we are already doing that kind of. Anyway, it's not needed for now.)
* **shuffle** is whether to shuffle it. Just leave this at ``True`` I think.
* **check** is some data check that's now deprecated. Just leave it as is.

## Distributions

Here, there are two distributions specified: one for conserved quantities and another for the time. These are represented vaguely the same way, using a pretty strong construct for generating distributions. Here are some basic types of distributions you can make:

You can find examples of each kind of distribution in the data configs already (mostly), so that might help with constructing your own. You can also ask me or look in the source code (the relevant bits are in ``distributions.sample_distribution``).

Here are some "primitive" distributions, that aren't combinations of other distributions:

* **Uniform**. Specify this with ``"type": "uniform"``, ``"dims": "[number of dimensions]"``, and then ``"min"`` should be a string that's a comma-separated list of the minimum of each dimension, and similar with ``max``. This is a uniform distribution between these mins/maxes (imagine a rectangular prism).
* **Uniform with intervals**. Like uniform, but you can have intervals cut out in the middle. On 1d, this is rather easy to imagine; for example, you might have the union of [0,0.33] and [0.67,1.00]. In more than 1D, there's two decently sensible way to do this. One is the ``"combine"="all"``, which creates rectangular prism cutouts within rectangular prisms; the other is ``"combine"="any"``, which makes the distribution itself a union of rectangular prisms. So if along axis 1 I want [0,0.33]+[0.67,1.00] and along axis 2 I want [0,0.2]+[0.8,1.0], for the first I'd get a square with a rectangular cutout in the middle, whereas for the second I'd get four rectangles at the corners of the square instead. To actually specify the intervals, there is ``even_space`` mode (set ``"mode"="even_space"``) which requires min/maxes specified like in Uniform, and then in ``"intervals"`` you specify the number of gaps that you want (integer at least 0) per dimension. Otherwise, you can use ``explicit`` mode in which you have to specify a comma-separated sequence of nested Python lists, each of which contains the allowable intervals along each axis (e.g. the previous examples would have ``"intervals"="[[0,0.33],[0.67,1.00]],[[0,0.2],[0.8,1.0]]"``). (Sorry this is long and probably kind of confusing; I can try to rewrite this if it's unclear.)
* **Single**. To specify a single value, just specify ``"type"="single"`` and then ``value`` should be a comma-separated list of the values that you want (which has length as long as the number of dimensions you want this vector to have.)
* **Discrete**. You can pick between an integer number of values. Specify probability of each with ``"ratio"``. Then the values should be specified as a 2D ``"values"``.
* **Exponential**. Exponential 1D distribution controlled by ``"scale"`` and ``"shift"``.

Here are distributions that are combinations of other distributions:
* **combine**. It picks between a number of sub distributions at random, and then samples from one of those, for each sample. Specify the probability of each subdistribution with ``"ratio"``, and then specify each subdistribtuion as element of a list that goes to ``"dists"``.
* **stack**. Each new distribution specified in ``"dists"`` creates new dimensions associated with it. If you have a 1D distribution, then a 2D distribution, then another 2D distribution, the first occupies dim 1, then the second dims 2 and 3, and the third dims 4 and5, so you get back 5D vectors. If it's not convenient for the generated dimensions to be right next each other, you can reorder them with ``reorder_axes``. This should be a list of the numbers 0 to D-1 (where D is the number of dimensions) that's a permutation of 0 to D-1, and 0 should go where the first dimension should go, for example, if I want to reverse 5 dimensions, I'd put ``"reorder_axes"="4,3,2,1,0".``

Hopefully this makes how the distribution generation works a bit easier to understand. This system is made to be hopefully as extensible as possible, so we can run all sorts of experiments.

## Other utility functions

There are a couple other useful utility functions exposed in ``ldcl.data``.
* ``ldcl.data.physics.combine_datasets`` allows you to generate a dataset that's a combination of other existing config files (that way you don't have to make a new config file just to merge the two.)
* ``ldcl.data.dists.is_in_distribution`` allows you to check if values are in a given data distribution.

These might be useful for some more advanced experiments.
