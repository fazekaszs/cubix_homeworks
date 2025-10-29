from pathlib import Path
import datetime

import pandas as pd

import matplotlib.pyplot as plt


DATA_PATH1 = Path("data/scrobbles-fazekaszs-1760773126.csv")
DATA_PATH2 = Path("data/scrobbles-Ada_gyenes-1760091268.csv")

WEEKDAYS = [
    "Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"
]

# Global pandas configurations
pd.set_option("display.max_columns", 10)  # in number of columns
pd.set_option("display.width", 200)  # in characters


def process_table_inplace(df: pd.DataFrame) -> pd.DataFrame:

    # Convert time string to DateTime object
    df["utc_time"] = df["utc_time"].map(
        lambda utc_t: datetime.datetime.strptime(utc_t, "%d %b %Y, %H:%M")
    )

    # Extract weekday
    df["weekday"] = df["utc_time"].map(
        lambda utc_t: WEEKDAYS[utc_t.weekday()]
    )

    # Extract morning/evening information
    df["before_noon"] = df["utc_time"].map(
        lambda utc_t: utc_t.hour < 12
    )

    # Extract only time (without date) in minutes
    df["day_minute"] = df["utc_time"].map(
        lambda utc_t: utc_t.hour * 60 + utc_t.minute,
    )

    # Create a full name identifier
    df["full_name"] = df["artist"].fillna("-") + " :: " + df["album"].fillna("-") + " :: " + df["track"].fillna("-")

    # Add a times listened counter
    times_listened = df["full_name"].value_counts()
    times_listened = times_listened.rename("times_listened")
    df = pd.merge(df, times_listened, how="left", on="full_name")

    # Drop unneeded columns
    df.drop(
        columns=["uts", "artist", "artist_mbid", "album", "album_mbid", "track", "track_mbid"],
        inplace=True
    )

    # Set new index
    df.set_index("utc_time", inplace=True)
    df.sort_index(inplace=True)
    df.index.names = ["time", ]

    return df


def int_to_time(x: int) -> datetime.time:

    time_hour, time_minute = divmod(x, 60)
    return datetime.time(
        hour=int(time_hour),
        minute=int(time_minute),
        second=int(60 * (time_minute - int(time_minute)))
    )


def main():

    df1_scrobbles: pd.DataFrame = pd.read_csv(DATA_PATH1)
    df1_scrobbles = process_table_inplace(df1_scrobbles)

    print(f">>> Music listening habits for person 1 (head):\n{df1_scrobbles.head()}", end="\n\n")

    print(f">>> Music listening habits for person 1 (iloc):\n{df1_scrobbles.iloc[-10:]}", end="\n\n")

    df2_scrobbles: pd.DataFrame = pd.read_csv(DATA_PATH2)
    df2_scrobbles = process_table_inplace(df2_scrobbles)

    print(f">>> Music listening habits for person 2 (head):\n{df2_scrobbles.head()}", end="\n\n")

    print(f">>> Music listening habits for person 2 (iloc):\n{df2_scrobbles.iloc[-10:]}", end="\n\n")

    pivot_table1 = df1_scrobbles.pivot_table(
        index=["weekday", "before_noon"],
        values=["times_listened", "day_minute"],
        aggfunc={"times_listened": ["mean", "std"], "day_minute": ["mean", "std"]}
    )
    pivot_table1[("day_minute", "mean")] = pivot_table1[("day_minute", "mean")].map(int_to_time)

    print(f">>> Pivot table for person 1:\n{pivot_table1}", end="\n\n")

    pivot_table2 = df2_scrobbles.pivot_table(
        index=["weekday", "before_noon"],
        values=["times_listened", "day_minute"],
        aggfunc={"times_listened": ["mean", "std"], "day_minute": ["mean", "std"]}
    )
    pivot_table2[("day_minute", "mean")] = pivot_table2[("day_minute", "mean")].map(int_to_time)

    print(f">>> Pivot table for person 2:\n{pivot_table2}", end="\n\n")

    joint_pivot_table = pd.merge(
        pivot_table1, pivot_table2,
        how="inner", left_index=True, right_index=True,
        suffixes=("_A", "_B")
    )

    print(f">>> The joint pivot table looks like this:\n{joint_pivot_table}", end="\n\n")

    joint_pivot_table.to_csv("joint_pivot_table.csv")

    melted_joint_pivot_table = pd.melt(
        joint_pivot_table,
        ignore_index=False,
        value_vars=[("times_listened_A", "mean"), ("times_listened_B", "mean")]
    )
    melted_joint_pivot_table.drop(columns=["variable_1", ], inplace=True)
    melted_joint_pivot_table.rename(columns={"variable_0": "variable"}, inplace=True)

    print(f">>> Melted joint pivot table for the times_listened columns:\n{melted_joint_pivot_table}", end="\n\n")

    print(f">>> Describe the melted joint pivot table using statistics:\n{melted_joint_pivot_table.describe()}")

    # Plotting
    fig, ax = plt.subplots(2, 2)
    fig.set_size_inches(12, 6)
    fig.subplots_adjust(top=0.85, hspace=0.6)
    fig.suptitle(
        "Summary of Music Listening Habits",
        fontweight="bold", fontsize="x-large"
    )

    df1_scrobbles["times_listened"].plot.hist(ax=ax[0, 0], bins=100)
    df2_scrobbles["times_listened"].plot.hist(ax=ax[1, 0], bins=100)
    ax[0, 0].set_title("Distribution of Times Listened\nof a Single Track for Person A")
    ax[1, 0].set_title("Distribution of Times Listened\nof a Single Track for Person B")
    ax[0, 0].set_xlabel("Times Listened")
    ax[1, 0].set_xlabel("Times Listened")

    df1_scrobbles["weekday"].map(lambda d: WEEKDAYS.index(d)).plot.hist(
        ax=ax[0, 1], range=(0, 7), bins=7,
        edgecolor="black", align="left", rwidth=0.5
    )
    df2_scrobbles["weekday"].map(lambda d: WEEKDAYS.index(d)).plot.hist(
        ax=ax[1, 1], range=(0, 7), bins=7,
        edgecolor="black", align="left", rwidth=0.5
    )
    ax[0, 1].set_xticks(list(range(7)), labels=WEEKDAYS)
    ax[1, 1].set_xticks(list(range(7)), labels=WEEKDAYS)
    ax[0, 1].set_title("Distribution of Listening Amounts\nat Different Weekdays for Person A")
    ax[1, 1].set_title("Distribution of Listening Amounts\nat Different Weekdays for Person B")
    ax[0, 1].set_xlabel("Weekday")
    ax[1, 1].set_xlabel("Weekday")

    fig.savefig("histograms.svg", dpi=300)


if __name__ == "__main__":
    main()
