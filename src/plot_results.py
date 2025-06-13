import matplotlib.pyplot as plt

def plot_results(df, list_values, filename, config):
    fig = df[list_values].plot().get_figure()
    fig.savefig(config.OUTPUT_PATH / filename)
    plt.close(fig)


def plot_clusters(path: pd.Series, title: str):
    """
        Plots all images within a cluster (or labeled as 'outliers')
    """

    # Calculate rows and columns for subplots
    items = len(path)
    num_cols = math.ceil(math.sqrt(items))
    num_rows = math.ceil(items / num_cols)

    # Define figure
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 10))
    axes = axes.flatten()
    fig.suptitle(title, fontsize=16)
    fig.subplots_adjust(top=0.95)

    # Read image
    for i, file in enumerate(path):
        image = cv2.imread(file)  # Read image
        image = cv2.resize(image, (224, 224))  # Resize image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
        axes[i].imshow(image)  # Plot image
        axes[i].axis('off')  # Remove axis
        i += 1

    # Hide any unused subplots
    for j in range(items, len(axes)):
        axes[j].axis('off')

    plt.show()