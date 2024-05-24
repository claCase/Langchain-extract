import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO
import os
import numpy as np


DATA_PATH = os.environ.get("DATA_PATH")


def select_file() -> str:
    _, _, files = next(os.walk(DATA_PATH))
    files = [file_ for file_ in files if file_.split(".")[-1] == "csv"]
    files.sort()
    files = files[::-1]
    selected = st.selectbox("Select a file", files)
    return selected


def make_bump_plot_total_topics(df, fig=None, ax=None):
    df["topics_count"] = df["Topics"].apply(lambda x: len(x.split(",")))
    summed = df.groupby("ID")["topics_count"].sum()
    if fig is None or ax is None:
        fig, ax = plt.subplots()
    ax.plot(summed.index, summed.values, color="red", label="Total Topics")
    ax.scatter(summed.index, summed.values, color="black")
    ax.set_xlabel("Chapter ID")
    ax.set_ylabel("Number of Topics")
    return fig, ax


def make_bump_plot_unique_topics(df, fig=None, ax=None):
    uid = df["ID"].unique()
    topics_count = {}
    for i in uid:
        topicls = []
        topics = df[df.ID == i]["Topics"].values.tolist()
        for t in topics:
            t_ = t.split(",")
            topicls.extend(t_)
        topics_count[i] = len(set(topicls))

    if fig is None or ax is None:
        fig, ax = plt.subplots()
    ax.plot(
        topics_count.keys(), topics_count.values(), color="blue", label="Unique Topics"
    )
    ax.scatter(topics_count.keys(), topics_count.values(), color="green")
    ax.set_xlabel("Chapter ID")
    ax.set_ylabel("Number of Topics")
    return fig, ax



def make_bump_plot_sa(df):
    scores = {"Negativo": -1, "Neutro": 0, "Positivo": 1}
    df["Score"] = df["Sentiment"].apply(lambda x: scores[x])
    mean_score = df.groupby(["ID"])["Score"].mean()
    fig, ax = plt.subplots()
    ax.plot(mean_score.index, mean_score.values)
    ax.scatter(mean_score.index, mean_score.values)
    ax.set_yticks(np.arange(-1, 2), list(scores.keys()))
    ax.set_xlabel("Chapter ID")
    ax.set_ylabel("Sentiment")
    fig.suptitle("Plot sentiment by chapter")
    return fig


if __name__ == "__main__":
    with st.status("Waiting for files..."):
        csv_files_found = False
        while not csv_files_found:
            try:
                _, _, files = next(os.walk(DATA_PATH))
                for file_ in files:
                    if file_.split(".")[-1] == "csv":
                        print("csv files found!!!")
                        csv_files_found = True
            except StopIteration:
                continue 
            

    selected = select_file()
    if selected is not None:
        df_path = os.path.join(DATA_PATH, selected)
        df = pd.read_csv(df_path, index_col=0)
        df["ID"] = df["ID"].astype(str)
        df.sort_values(by="ID", inplace=True)
        st.dataframe(df)
        fig, ax = make_bump_plot_unique_topics(df)
        fig, ax = make_bump_plot_total_topics(df, fig, ax)
        fig.suptitle("Plot number of topics")
        ax.legend()
        fig2 = make_bump_plot_sa(df)
        st.pyplot(fig)
        st.pyplot(fig2)
