import numpy as np
import altair as alt
import pandas as pd
import streamlit as st

st.header('st.write')
st.markdown(":green[$\sqrt{x^2+y^2+z}=1$] is a Pythagorean identity. :pencil:")
st.markdown("This text is :red[colored red], and this is **:blue[colored]** and bold.")

code = '''def hello():
    print("Hello, Streamlit!")'''
st.code(code, language='python')


# Example 1
st.write('Hello, *World!* :sunglasses:')

# Example 2
st.write(1234)

# Example 3
df = pd.DataFrame({
     'first column': [1, 2, 3, 4],
     'second column': [10, 20, 30, 40]
     })
st.write(df)

# Example 4
st.write('Below is a DataFrame:', df, 'Above is a dataframe.')

# Example 5
df2 = pd.DataFrame(
     np.random.randn(200, 3),
     columns=['a', 'b', 'c'])

st.subheader('table')
st.write(df2)

c = alt.Chart(df2).mark_circle().encode(
     x='a', y='b', size='c', color='c', tooltip=['a', 'b', 'c'])
st.subheader('chart')
st.write(c)