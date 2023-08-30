import streamlit as st
# import pdf_reader
import embeddings_function
# import call_function


st.markdown("# Embed  ❄️")
st.sidebar.markdown("# Embed  ❄️")

if st.button("Embed"):
    st.write(embeddings_function.pdf_embed())





# if st.button("Embed"):
#     function1 = pdf_reader.pdf_embed()
#     st.write(call_function.pdf_embed())
#     st.write(pdf_reader.pdf_embed())