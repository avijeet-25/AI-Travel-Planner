import streamlit as st

if 'grocery_items' not in st.session_state:
     st.session_state.grocery_items = ['Apple', 'Banana', 'Carrot', 'Milk', 'Eggs']

st.title('Grocery List App')

new_item = st.text_input("Add a new item to your grocery list:")

if st.button("Add Item"):
     if new_item:
          st.session_state.grocery_items.append(new_item)
          st.success(f"'{new_item}' has been added to your list!")
     else:
          st.warning("Please enter an item to add.")

st.write('### Items to Buy:')
for item in st.session_state.grocery_items:
     st.write(f"-{item}")