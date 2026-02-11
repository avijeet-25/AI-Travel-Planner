import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("Simple streamlit App with Box plot")

data = {
    'Category': ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C'],
    'Values': [10, 20, 15, 25, 30, 20, 35, 40, 45]
}

df = pd.DataFrame(data)

st.write('Here is the sample DataFrame:')
st.dataframe(df.style.highlight_max(axis=0))

fig, ax = plt.subplots()
df.boxplot(column='Values', by='Category', ax=ax, grid=False)
plt.title('Box plot of values by category')
plt.suptitle("")
plt.xlabel('Category')
plt.ylabel('Values')
st.pyplot(fig)

city_data = {
    'City': ['Palermo', 'Syracuse', 'Catania', 'Agrigento'],
    'latitude': [38.1157, 37.0757, 37.5079, 37.2982],
    'longitude': [13.3615, 15.2867, 15.0830, 13.5763]
}

city_data = pd.DataFrame(city_data)
st.map(city_data)