import matplotlib.pyplot as plt
import pandas as pd

import src.config as config
from src.helpers import save_plot


def visualise_hog(HoG) -> None:
    if config.verbose_mode:
        print("\n{} training set HoG features:".format(config.dataset))
        print(HoG.head(5))
        print(HoG.info())

    fig = plt.figure(figsize=(10, 10))

    image_background1 = HoG.iloc[0].to_numpy().reshape(30, 30)
    fig.add_subplot(221)
    plt.imshow(image_background1, cmap="bone")
    plt.title("background", fontsize=config.fontsizes['title'])
    plt.axis("off")

    image_background2 = HoG.iloc[1080].to_numpy().reshape(30, 30)
    fig.add_subplot(222)
    plt.imshow(image_background2, cmap="bone")
    plt.title("background", fontsize=config.fontsizes['title'])
    plt.axis("off")

    image_seal1 = HoG.iloc[62209].to_numpy().reshape(30, 30)
    fig.add_subplot(223)
    plt.imshow(image_seal1, cmap="bone")
    plt.title("seal", fontsize=config.fontsizes['title'])
    plt.axis("off")

    image_seal2 = HoG.iloc[62100].to_numpy().reshape(30, 30)
    fig.add_subplot(224)
    plt.imshow(image_seal2, cmap="bone")
    plt.title("seal", fontsize=config.fontsizes['title'])
    plt.axis("off")

    save_plot("reconstructed_images_binary")
    plt.show()


def visualise_rgb_hist(RGB_hist):
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
    plt.title("RGB Histogram for a single image", fontsize=config.fontsizes['title'])
    plt.grid(color='#95a5a6', linestyle='-', linewidth=1, axis='y', alpha=0.5)
    plt.legend()
    save_plot("rgb_hist_single_image")
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
    plt.title("Average RGB Histogram of all images in the training set", fontsize=config.fontsizes['title'])
    plt.grid(color='#95a5a6', linestyle='-', linewidth=1, axis='y', alpha=0.5)
    plt.legend()
    save_plot("rgb_avg_hist_all_training_set")
    plt.show()


def visualise_class_distribution(outputs):
    if config.verbose_mode:
        print("\n{} training set class distribution:".format(config.dataset))
        outputs.info()

    total_occurrences = outputs.shape[0]
    class_distribution = {
        'seal': {
            'occurrences': outputs[0].value_counts()["seal"],
            'distribution': round(outputs[0].value_counts()["seal"] / total_occurrences, 5)
        },
        'background': {
            'occurrences': outputs[0].value_counts()["background"],
            'distribution': round(outputs[0].value_counts()["background"] / total_occurrences, 5)
        }
    }
    print(class_distribution)

    # Data preparation for the bar chart.
    data = [class_distribution['seal']['occurrences'], class_distribution['background']['occurrences']]
    value_labels = [class_distribution['seal']['distribution'], class_distribution['background']['distribution']]
    x_axis_labels = ["seal", "background"]

    # Bar chart.
    plt.figure(figsize=(10, 8))
    plt.xticks(range(len(data)), x_axis_labels)
    plt.xlabel("Class", fontsize=config.fontsizes['axis'])
    plt.ylabel("Occurrences", fontsize=config.fontsizes['axis'])
    plt.title("Binary dataset: Class frequency occurrences & distribution", fontsize=config.fontsizes['title'])
    plt.bar(x_axis_labels, data, color='steelblue')
    plt.grid(color='#95a5a6', linestyle='-', linewidth=1, axis='y', alpha=0.5)

    # Add value labels for each bar.
    for a, b in zip(x_axis_labels, data):
        plt.text(a, b, "{}%".format(str(round(class_distribution[a]['distribution'] * 100, 2))), fontsize=15)

    # Save and display chart.
    save_plot("binary_class_distribution")
    plt.show()


def visualise_correlation(X, y):
    temp_data = pd.concat([X, y], axis=1)
    correlation_matrix = temp_data.corr()

    if config.verbose_mode:
        print("\n{} training set correlation matrix:".format(config.dataset))
        print(correlation_matrix)

    print("10 features with the strongest positive correlation with the  output:")
    print(correlation_matrix.iloc[963].sort_values(ascending=False).head(11))
    print("\n10 features with the strongest negative correlation with the  output:")
    print(correlation_matrix.iloc[963].sort_values(ascending=True).head(10))
