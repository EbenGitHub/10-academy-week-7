import re
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import font_manager


def plot_channel_distribution(df):
    """
    Plots a donut chart and horizontal bar graph to visualize the distribution of messages by channel.
    """
    # Get value counts
    channel_counts = df["channel_address"].value_counts()

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Define color palette
    colors = sns.color_palette("magma", len(channel_counts))

    # Donut Chart
    wedges, texts, autotexts = axes[0].pie(
        channel_counts,
        labels=channel_counts.index,
        autopct="%1.1f%%",
        startangle=140,
        colors=colors,
        wedgeprops={"edgecolor": "white"},
    )
    axes[0].add_artist(plt.Circle((0, 0), 0.5, color="white"))
    axes[0].set_title("Distribution of Messages by Channel (Donut Chart)")

    # Horizontal Bar Graph
    sns.barplot(
        y=channel_counts.index, x=channel_counts.values, palette="magma", ax=axes[1]
    )
    axes[1].set_xlabel("Message Count")
    axes[1].set_ylabel("Channel Address")
    axes[1].set_title("Messages per Channel (Horizontal Bar Graph)")

    # Add count labels on bars
    for index, value in enumerate(channel_counts.values):
        axes[1].text(
            value + 130, index, str(value), va="center", fontsize=13
        )  # Adjust position slightly

    # Adjust layout
    plt.tight_layout()
    plt.show()


def plot_gantt_chart(df):
    """
    Plots a Gantt chart for each channel's message start and end dates.
    Uses month and year on the x-axis.
    """
    # Convert 'date' column to datetime
    df["date"] = pd.to_datetime(df["date"])

    # Get start and end dates for each channel
    channel_dates = (
        df.groupby("channel_address")["date"].agg(["min", "max"]).reset_index()
    )
    channel_dates.columns = ["channel_address", "start_date", "end_date"]

    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 7))

    # Plot each channel as a horizontal bar with the specified color
    for index, row in channel_dates.iterrows():
        ax.barh(
            row["channel_address"],
            (row["end_date"] - row["start_date"]).days,
            left=row["start_date"],
            height=0.6,
            color="#C4AD9D",
            edgecolor="grey",
        )

    # Formatting the x-axis to show month and year
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.xticks(rotation=90)  # Make x-axis labels vertical

    # Add vertical grid lines
    ax.grid(True, axis="x", linestyle="--", alpha=0.7)

    # Labels and title
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Channel Address", fontsize=12)
    plt.title("Message Start and End Dates for Each Channel", fontsize=14)

    # Show the plot
    plt.tight_layout()
    plt.show()


def generate_hist_box_plots(df, plot_data):
    """Generates histplots and boxplots for a list of dictionaries of title and column name."""
    num_cols = len(plot_data)
    fig, axes = plt.subplots(
        nrows=2,
        ncols=num_cols,
        figsize=(14, 5),
        # sharey="row",
        sharex="col",
        gridspec_kw={"height_ratios": [7, 0.4]},
    )

    for i in range(len(plot_data)):
        sns.histplot(df[plot_data[i]["column"]], kde=True, ax=axes[0, i])
        sns.boxplot(x=df[plot_data[i]["column"]], ax=axes[1, i], orient="h")
        axes[1, i].set_xlabel(plot_data[i]["label"], fontsize=13)
        axes[0, i].set_title(plot_data[i]["title"], fontsize=13)

    axes[0, 0].set_ylabel("Frequency", fontsize=13)
    axes[0, 1].set_ylabel("Frequency", fontsize=13)
    fig.tight_layout()
    plt.show()


def plot_word_cloud(df, language=None):
    """
    Plots a word cloud for the 'cleaned_message' column based on the specified language.
    """
    # Regular expression for Amharic characters (U+1200 to U+137F)
    amharic_pattern = re.compile(r"[\u1200-\u137F]+")
    english_pattern = re.compile(r"[A-Za-z]+")

    # Concatenate all the text from 'cleaned_message' column
    all_text = " ".join(df["cleaned_message"].dropna())

    if language == "amharic":
        # Exclude English words when plotting for Amharic
        amharic_text = " ".join(re.findall(r"[\u1200-\u137F]+", all_text))
        font_path = "../assets/fonts/NotoSerifEthiopic_Condensed-Regular.ttf"  # Path to the Amharic font
        text_to_plot = amharic_text
    elif language == "english":
        # Exclude Amharic words when plotting for English
        english_text = " ".join(re.findall(r"[A-Za-z]+", all_text))
        font_path = None  # Default font for English
        text_to_plot = english_text
    else:
        # Use all text if no language specified
        text_to_plot = all_text
        font_path = "../assets/fonts/NotoSerifEthiopic_Condensed-Regular.ttf"  # Path to the Amharic font

    # Generate the word cloud
    wordcloud = WordCloud(
        width=800, height=400, background_color="white", font_path=font_path
    ).generate(text_to_plot)

    # Plot the word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")  # Hide axes
    plt.title(
        f"Word Cloud for {'All' if language is None else language.capitalize()} Messages"
    )
    plt.show()


def plot_top_tokens(df, top_n=20):
    """
    Plot the top N most frequent tokens in a horizontal bar chart using the given font.

    Parameters:
    - df: DataFrame with a 'token' column containing the tokens
    - top_n: The number of top frequent tokens to display (default is 20)
    - font_path: Path to the Amharic font file (default is set to a specific path)
    """
    # Load the font
    font_path = "/content/drive/MyDrive/10 acadamy/W5 Challenge/assets/Untitled Folder/NotoSerifEthiopic_Condensed-Regular.ttf"
    prop = font_manager.FontProperties(fname=font_path)

    # Get the top N most frequent tokens
    top_tokens = df["token"].value_counts().head(top_n)

    # Create a horizontal bar chart
    plt.figure(figsize=(8, 4))
    sns.barplot(x=top_tokens.values, y=top_tokens.index, palette="magma")

    # Set the font for the entire plot, including labels, ticks, and title
    plt.xlabel("Frequency", fontproperties=prop)
    plt.ylabel("Token", fontproperties=prop)
    plt.title(f"Top {top_n} Most Frequent Tokens", fontproperties=prop)

    # Update tick labels to use the Amharic font as well
    plt.xticks(fontproperties=prop)
    plt.yticks(fontproperties=prop)
    plt.show()


def plot_label_distribution(df):
    """
    Plot the distribution of labels in a bar chart using the magma color palette.

    Parameters:
    - df: DataFrame with a 'label' column
    """
    # Get the label distribution
    label_counts = df["label"].value_counts()

    # Create a bar chart
    plt.figure(figsize=(8, 4))
    sns.barplot(x=label_counts.index, y=label_counts.values, palette="magma")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.title("Label Distribution")
    plt.show()
