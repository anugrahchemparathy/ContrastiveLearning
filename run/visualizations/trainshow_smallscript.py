import imageio

images = []
for i in range(200):
    print(i)
    images.append(imageio.imread(f'train_progression/{i}e.png'))
imageio.mimsave('training_s.gif', images, fps=6)
