import streamlit as st
conn = st.connection('my_database_sql')
df = conn.query("select * from my_beautiful_table")
st.dataframe(df)