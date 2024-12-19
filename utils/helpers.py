import PIL.Image
import torchvision.transforms as T
import matplotlib.pyplot as plt

def plot_dataset(images, truths, single=False):
    """
    Plots images alongside their corresponding ground-truth labels.

    @param images: A list of images or a single image to be plotted.
    @param truths: A list of ground-truth labels corresponding to the images.
    @param single: If True, plots a single image and its label. Otherwise, plots all images and labels.

    @return None: Displays the plots.
    """
    if not single:
        for image, label in zip(images, truths):
            plot(image, label)
    else:
        print(images)
        if type(images) == PIL.Image.Image:
            transform = T.Compose([T.PILToTensor()])
            images = transform(images)
        if type(truths) == PIL.Image.Image:
            transform = T.Compose([T.PILToTensor()])
            truths = transform(truths)

        plot(images,truths)

def plot(img, lbl):
    """
        Plots an image alongside its corresponding ground-truth labels.

        @param img: Image to be plotted.
        @param lbl: Label corresponding to the image.

        @return None: Displays the plots.
        """
    plt.subplot(1, 2, 1)
    plt.imshow(img.permute(1, 2, 0).cpu().numpy())
    plt.axis('off')
    plt.title("Image")

    # Plot the label
    plt.subplot(1, 2, 2)
    plt.imshow(lbl.permute(1, 2, 0).cpu().numpy())
    plt.title("Label")

    plt.tight_layout()
    plt.show()

def plot_3_images(image, label, blended):
    """
        Plots an image, its corresponding ground-truth label, and a blended overlay side-by-side.

        @param image: The original image.
        @param label: The ground-truth label image.
        @param blended: An image showing a blend of the original and label.

        @return:None: Displays the three images in a single figure.
        """
    transform = T.Compose([T.PILToTensor()])

    if type(image) == PIL.Image.Image:
        image = transform(image)
    if type(label) == PIL.Image.Image:
        label = transform(label)
    if type(blended) == PIL.Image.Image:
        blended = transform(blended)

    plt.figure(figsize=(15, 5))  # Set figure size

    # Plot the first image
    plt.subplot(1, 3, 1)
    plt.imshow(image.permute(1, 2, 0).cpu().numpy())
    plt.axis('off')
    plt.title("Image")

    # Plot the second image
    plt.subplot(1, 3, 2)
    plt.imshow(label.permute(1, 2, 0).cpu().numpy())
    plt.axis('off')
    plt.title("Label")

    # Plot the third image
    plt.subplot(1, 3, 3)
    plt.imshow(blended.permute(1, 2, 0).cpu().numpy())
    plt.axis('off')
    plt.title("Blended")

    plt.tight_layout()
    plt.show()
