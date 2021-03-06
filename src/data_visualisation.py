import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd

import config as config
from helpers import save_plot


def data_overview(dataset) -> None:
    """
    Call DataFrame overview functions.
    :param dataset:
    :return: None.
    """
    print(dataset.info())
    print(dataset.describe())


def visualise_hog(HoG) -> None:
    """
    Reshape the HoG features into
    :param HoG:
    :return: None.
    """
    if config.verbose_mode:
        print("\n{} training set HoG features:".format(config.dataset))
        print(HoG.head(5))
        print(HoG.info())

    if config.dataset == "binary":
        fig = plt.figure(figsize=(10, 10))

        image_bg1 = mpimg.imread("../data/exampleImages/bg1.png")
        fig.add_subplot(2, 3, 1)
        plt.imshow(image_bg1, cmap="bone")
        plt.title("actual background", fontsize=config.fontsizes['title'])
        plt.axis("off")

        image_background1 = HoG.iloc[0].to_numpy().reshape(30, 30)
        fig.add_subplot(2, 3, 2)
        plt.imshow(image_background1, cmap="bone")
        plt.title("reconstructed background", fontsize=config.fontsizes['title'])
        plt.axis("off")

        image_background2 = HoG.iloc[1080].to_numpy().reshape(30, 30)
        fig.add_subplot(2, 3, 3)
        plt.imshow(image_background2, cmap="bone")
        plt.title("reconstructed background", fontsize=config.fontsizes['title'])
        plt.axis("off")

        image_whitecoat2 = mpimg.imread("../data/exampleImages/whitecoat2.png")
        fig.add_subplot(2, 3, 4)
        plt.imshow(image_whitecoat2, cmap="bone")
        plt.title("actual background", fontsize=config.fontsizes['title'])
        plt.axis("off")

        image_seal1 = HoG.iloc[62209].to_numpy().reshape(30, 30)
        fig.add_subplot(2, 3, 5)
        plt.imshow(image_seal1, cmap="bone")
        plt.title("reconstructed seal", fontsize=config.fontsizes['title'])
        plt.axis("off")

        image_seal2 = HoG.iloc[62100].to_numpy().reshape(30, 30)
        fig.add_subplot(2, 3, 6)
        plt.imshow(image_seal2, cmap="bone")
        plt.title("reconstructed seal", fontsize=config.fontsizes['title'])
        plt.axis("off")

    elif config.dataset == "multi":
        fig = plt.figure(figsize=(20, 8))

        image_background1 = HoG.iloc[7778].to_numpy().reshape(30, 30)
        fig.add_subplot(3, 5, 1)
        plt.imshow(image_background1, cmap="bone")
        plt.title("background", fontsize=config.fontsizes['title'])
        plt.axis("off")

        image_whitecoat1 = HoG.iloc[0].to_numpy().reshape(30, 30)
        fig.add_subplot(3, 5, 2)
        plt.imshow(image_whitecoat1, cmap="bone")
        plt.title("whitecoat", fontsize=config.fontsizes['title'])
        plt.axis("off")

        image_moultedpup1 = HoG.iloc[4981].to_numpy().reshape(30, 30)
        fig.add_subplot(3, 5, 3)
        plt.imshow(image_moultedpup1, cmap="bone")
        plt.title("moulted pup", fontsize=config.fontsizes['title'])
        plt.axis("off")

        image_deadpup1 = HoG.iloc[7253].to_numpy().reshape(30, 30)
        fig.add_subplot(3, 5, 4)
        plt.imshow(image_deadpup1, cmap="bone")
        plt.title("dead pup", fontsize=config.fontsizes['title'])
        plt.axis("off")

        image_juvenile1 = HoG.iloc[7530].to_numpy().reshape(30, 30)
        fig.add_subplot(3, 5, 5)
        plt.imshow(image_juvenile1, cmap="bone")
        plt.title("juvenile", fontsize=config.fontsizes['title'])
        plt.axis("off")

        image_background2 = HoG.iloc[62205].to_numpy().reshape(30, 30)
        fig.add_subplot(3, 5, 6)
        plt.imshow(image_background2, cmap="bone")
        plt.title("background", fontsize=config.fontsizes['title'])
        plt.axis("off")

        image_whitecoat2 = HoG.iloc[4976].to_numpy().reshape(30, 30)
        fig.add_subplot(3, 5, 7)
        plt.imshow(image_whitecoat2, cmap="bone")
        plt.title("whitecoat", fontsize=config.fontsizes['title'])
        plt.axis("off")

        image_moultedpup2 = HoG.iloc[7248].to_numpy().reshape(30, 30)
        fig.add_subplot(3, 5, 8)
        plt.imshow(image_moultedpup2, cmap="bone")
        plt.title("moulted pup", fontsize=config.fontsizes['title'])
        plt.axis("off")

        image_deadpup2 = HoG.iloc[7529].to_numpy().reshape(30, 30)
        fig.add_subplot(3, 5, 9)
        plt.imshow(image_deadpup2, cmap="bone")
        plt.title("dead pup", fontsize=config.fontsizes['title'])
        plt.axis("off")

        image_juvenile2 = HoG.iloc[7777].to_numpy().reshape(30, 30)
        fig.add_subplot(3, 5, 10)
        plt.imshow(image_juvenile2, cmap="bone")
        plt.title("juvenile", fontsize=config.fontsizes['title'])
        plt.axis("off")

        image_bg1 = mpimg.imread("../data/exampleImages/bg1.png")
        fig.add_subplot(3, 5, 11)
        plt.imshow(image_bg1, cmap="bone")
        plt.title("actual background", fontsize=config.fontsizes['title'])
        plt.axis("off")

        image_whitecoat1 = mpimg.imread("../data/exampleImages/whitecoat1.png")
        fig.add_subplot(3, 5, 12)
        plt.imshow(image_whitecoat1, cmap="bone")
        plt.title("actual whitecoat", fontsize=config.fontsizes['title'])
        plt.axis("off")

        image_moultedpup1 = mpimg.imread("../data/exampleImages/moultedpup1.png")
        fig.add_subplot(3, 5, 13)
        plt.imshow(image_moultedpup1, cmap="bone")
        plt.title("actual moulted pup", fontsize=config.fontsizes['title'])
        plt.axis("off")

        image_deadpup2 = mpimg.imread("../data/exampleImages/deadpup2.png")
        fig.add_subplot(3, 5, 14)
        plt.imshow(image_deadpup2, cmap="bone")
        plt.title("actual dead pup", fontsize=config.fontsizes['title'])
        plt.axis("off")

        image_juvenile3 = mpimg.imread("../data/exampleImages/juvenile3.png")
        fig.add_subplot(3, 5, 15)
        plt.imshow(image_juvenile3, cmap="bone")
        plt.title("actual juvenile", fontsize=config.fontsizes['title'])
        plt.axis("off")

    save_plot("reconstructed_images_{}".format(config.dataset))
    plt.show()


def visualise_rgb_hist(RGB_hist) -> None:
    """

    :param RGB_hist:
    :return:
    """
    if config.verbose_mode:
        print("\n{} training set RGB Histogram features:".format(config.dataset))
        print(RGB_hist.shape)
        print(RGB_hist.head(5))

    # Split channels.
    red_channel = RGB_hist.iloc[:, :16]
    blue_channel = RGB_hist.iloc[:, 16:32]
    green_channel = RGB_hist.iloc[:, 32:]

    # Visualise RGB histogram for a single image.
    plt.figure(figsize=(10, 8))
    bin_labels = [i for i in range(0, 255, 16)]
    plt.plot(bin_labels, red_channel.iloc[0].values, color="red", label="R")
    plt.plot(bin_labels, green_channel.iloc[0].values, color="green", label="G")
    plt.plot(bin_labels, blue_channel.iloc[0].values, color="blue", label="B")
    plt.xlim((0, 255 - 16))
    plt.ylim(0, max(max(red_channel.iloc[0].values),
                    max(green_channel.iloc[0].values),
                    max(blue_channel.iloc[0].values)) + 25)  # Add padding.
    plt.xlabel("Bins", fontsize=config.fontsizes['axis'])
    plt.ylabel("Pixel Frequency", fontsize=config.fontsizes['axis'])
    plt.title("RGB Histogram for a single image in {} dataset".format(config.dataset),
              fontsize=config.fontsizes['title'])
    plt.grid(color='#95a5a6', linestyle='-', linewidth=1, axis='y', alpha=0.5)
    plt.legend()
    save_plot("rgb_hist_single_image_{}".format(config.dataset))
    plt.show()

    # Visualise the aggregate RGB histogram for all images.
    plt.figure(figsize=(10, 8))
    plt.plot(bin_labels, red_channel.mean().values, color="red", label="R")
    plt.plot(bin_labels, green_channel.mean().values, color="green", label="G")
    plt.plot(bin_labels, blue_channel.mean().values, color="blue", label="B")
    plt.xlim((0, 255 - 16))
    plt.ylim(0, max(max(red_channel.mean().values),
                    max(green_channel.mean().values),
                    max(blue_channel.mean().values)) + 25)  # Add padding.
    plt.xlabel("Bins", fontsize=config.fontsizes['axis'])
    plt.ylabel("Average Pixel Frequency", fontsize=config.fontsizes['axis'])
    plt.title("Average RGB Histogram of all images in {} dataset".format(config.dataset),
              fontsize=config.fontsizes['title'])
    plt.grid(color='#95a5a6', linestyle='-', linewidth=1, axis='y', alpha=0.5)
    plt.legend()
    save_plot("{}_rgb_avg_hist_all_training_set".format(config.dataset))
    plt.show()


def visualise_class_distribution(outputs) -> None:
    """

    :param outputs:
    :return:
    """
    if config.verbose_mode:
        print("\n{} training set class distribution:".format(config.dataset))
        outputs.info()

    total_occurrences = outputs.shape[0]

    # Get the number of occurrences and % distribution of each class.
    class_distribution = dict()
    if config.dataset == "binary":
        class_distribution = {
            'background': {
                'occurrences': outputs[0].value_counts()["background"],
                'distribution': round(outputs[0].value_counts()["background"] / total_occurrences, 5)
            },
            'seal': {
                'occurrences': outputs[0].value_counts()["seal"],
                'distribution': round(outputs[0].value_counts()["seal"] / total_occurrences, 5)
            }
        }
    elif config.dataset == "multi":
        class_distribution = {
            'whitecoat': {
                'occurrences': outputs[0].value_counts()["whitecoat"],
                'distribution': round(outputs[0].value_counts()["whitecoat"] / total_occurrences, 5)
            },
            'background': {
                'occurrences': outputs[0].value_counts()["background"],
                'distribution': round(outputs[0].value_counts()["background"] / total_occurrences, 5)
            },
            'dead pup': {
                'occurrences': outputs[0].value_counts()["dead pup"],
                'distribution': round(outputs[0].value_counts()["dead pup"] / total_occurrences, 5)
            },
            'juvenile': {
                'occurrences': outputs[0].value_counts()["juvenile"],
                'distribution': round(outputs[0].value_counts()["juvenile"] / total_occurrences, 5)
            },
            'moulted pup': {
                'occurrences': outputs[0].value_counts()["moulted pup"],
                'distribution': round(outputs[0].value_counts()["moulted pup"] / total_occurrences, 5)
            },
        }
    print(class_distribution)

    # Data preparation for the bar chart.
    data = list()
    x_axis_labels = list()
    if config.dataset == "binary":
        data = [
            class_distribution['background']['occurrences'],
            class_distribution['seal']['occurrences']
        ]
        x_axis_labels = ["background", "seal"]
    elif config.dataset == "multi":
        data = [
            class_distribution['background']['occurrences'],
            class_distribution['whitecoat']['occurrences'],
            class_distribution['moulted pup']['occurrences'],
            class_distribution['dead pup']['occurrences'],
            class_distribution['juvenile']['occurrences']
        ]
        x_axis_labels = ["background", "whitecoat", "moulted pup", "dead pup", "juvenile"]

    # Bar chart.
    plt.figure(figsize=(10, 8))
    plt.xticks(range(len(data)), x_axis_labels)
    plt.xlabel("Class", fontsize=config.fontsizes['axis'])
    plt.ylabel("Occurrences", fontsize=config.fontsizes['axis'])
    plt.title("{} dataset: Class frequency occurrences & distribution".format(config.dataset),
              fontsize=config.fontsizes['title'])
    plt.bar(x_axis_labels, data, color='steelblue')
    plt.grid(color='#95a5a6', linestyle='-', linewidth=1, axis='y', alpha=0.5)

    # Add value labels for each bar.
    for a, b in zip(x_axis_labels, data):
        plt.text(a, b, "{}%".format(str(round(class_distribution[a]['distribution'] * 100, 2))), fontsize=15)

    # Save and display chart.
    save_plot("{}_class_distribution".format(config.dataset))
    plt.show()


def visualise_correlation(X, y) -> None:
    temp_data = pd.concat([X, y], axis=1)
    correlation_matrix = temp_data.corr()

    if config.verbose_mode:
        print("\n{} training set correlation matrix:".format(config.dataset))
        print(correlation_matrix)

    print("10 features with the strongest positive correlation with the class labels:")
    print(correlation_matrix.iloc[963].sort_values(ascending=False).head(11))
    print("\n10 features with the strongest negative correlation with the class labels:")
    print(correlation_matrix.iloc[963].sort_values(ascending=True).head(11))
