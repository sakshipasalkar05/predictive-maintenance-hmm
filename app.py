import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Engine Health Monitor", layout="wide")
st.title("Predictive Maintenance with a Hidden Markov Model")

art = joblib.load("models/artifacts.joblib")
model, scaler = art["model"], art["scaler"]
useful, features, state_rul = art["useful"], art["features"], art["state_rul"]

STATE_NAMES = ["Healthy", "Early wear", "Mild wear",
               "Moderate wear", "Severe wear", "Critical"]

cols = ["engine", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]

@st.cache_data
def load_test():
    df = pd.read_csv("data/test_FD001.txt", sep=r"\s+", header=None, names=cols)
    df[useful] = scaler.transform(df[useful])
    return df

test = load_test()

engine = st.sidebar.selectbox("Engine", sorted(test["engine"].unique()))
g = test[test["engine"] == engine].reset_index(drop=True)
cycle = st.sidebar.slider("Current cycle", 1, len(g), len(g))

Xe = g[features].values[:cycle]
post = model.predict_proba(Xe)[-1]
warn = post[4] + post[5]
rul_est = post @ state_rul

c1, c2, c3 = st.columns(3)
c1.metric("Most likely state", STATE_NAMES[int(post.argmax())])
c2.metric("Failure warning probability", f"{warn:.0%}")
c3.metric("Estimated RUL (cycles)", f"{rul_est:.0f}")

if warn >= 0.5:
    st.error("WARNING: engine is in the danger zone. Schedule maintenance.")
else:
    st.success("Engine operating normally.")

left, right = st.columns(2)

with left:
    st.subheader("State probabilities")
    fig, ax = plt.subplots()
    ax.bar(STATE_NAMES, post)
    ax.set_ylim(0, 1)
    plt.xticks(rotation=30)
    st.pyplot(fig)

with right:
    st.subheader("Sensor s11 so far")
    fig, ax = plt.subplots()
    ax.plot(g["cycle"][:cycle], g["s11"][:cycle])
    ax.set_xlabel("cycle")
    st.pyplot(fig)